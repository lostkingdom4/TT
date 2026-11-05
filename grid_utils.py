"""
Grid calculation utilities for Triton kernel execution.

Provides functions to compute grid dimensions based on problem size (numel values)
and block sizes (XBLOCK, YBLOCK, R0_BLOCK).
"""

from typing import Tuple, Dict, List, Optional
import os, io, re
import ast
import logging


def ceil_div(a: int, b: int) -> int:
    """
    Compute ceiling division: ceil(a / b) = (a + b - 1) // b

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Ceiling of a / b
    """
    return (a + b - 1) // b


def compute_grid_from_numel(
    numel_dict: Dict[str, int],
    block_config: Dict[str, int],
    kernel_type: Optional[str] = None
) -> Tuple[int, ...]:
    """
    Compute grid dimensions based on numel values and block configuration.

    This function determines the appropriate grid based on the kernel's dimensional
    characteristics (1D, 2D, or reduction) and computes the grid size using
    ceiling division to ensure all elements are processed.

    Args:
        numel_dict: Dictionary mapping numel parameter names to their values
                   (e.g., {'xnumel': 16384, 'ynumel': 256, 'r0_numel': 64})
        block_config: Dictionary with block size configuration
                     (e.g., {'XBLOCK': 256, 'YBLOCK': 32, 'R0_BLOCK': 2048})
        kernel_type: Optional kernel type hint ('1d', '2d', 'reduction', 'reduction_fixed')
                    If None, will be inferred from numel_dict and block_config

    Returns:
        Tuple of grid dimensions (1D, 2D, or 3D)

    Examples:
        >>> # 1D kernel
        >>> compute_grid_from_numel({'xnumel': 16384}, {'XBLOCK': 256})
        (64,)

        >>> # 2D kernel
        >>> compute_grid_from_numel({'xnumel': 1024, 'ynumel': 512}, {'XBLOCK': 32, 'YBLOCK': 32})
        (32, 16)

        >>> # Reduction kernel
        >>> compute_grid_from_numel({'xnumel': 256, 'r0_numel': 2048}, {'XBLOCK': 1, 'R0_BLOCK': 2048})
        (256,)

    Raises:
        ValueError: If required numel or block parameters are missing
    """
    # Extract numel values
    xnumel = numel_dict.get('xnumel')
    ynumel = numel_dict.get('ynumel')
    r0_numel = numel_dict.get('r0_numel')

    # Extract block sizes
    XBLOCK = block_config.get('XBLOCK')
    YBLOCK = block_config.get('YBLOCK')
    R0_BLOCK = block_config.get('R0_BLOCK')

    # Validate XBLOCK is present
    if XBLOCK is None:
        raise ValueError("XBLOCK not found in block configuration")

    if xnumel is None:
        raise ValueError("xnumel not found in numel_dict")

    # Infer kernel type if not provided
    if kernel_type is None:
        kernel_type = _infer_kernel_type(numel_dict, block_config)

    # Compute grid based on kernel type
    if kernel_type == '2d':
        # 2D kernel: requires both XBLOCK/YBLOCK and xnumel/ynumel
        if YBLOCK is None:
            raise ValueError("2D kernel requires YBLOCK in block configuration")
        if ynumel is None:
            raise ValueError("2D kernel requires ynumel in numel_dict")

        grid_x = ceil_div(xnumel, XBLOCK)
        grid_y = ceil_div(ynumel, YBLOCK)
        return (grid_x, grid_y)

    elif kernel_type in ['reduction', 'reduction_fixed']:
        # Reduction kernel: grid based on xnumel/XBLOCK
        # R0_BLOCK controls reduction dimension but doesn't affect grid size
        grid_x = ceil_div(xnumel, XBLOCK)
        return (grid_x,)

    elif kernel_type == '1d':
        # 1D kernel: simple grid based on xnumel/XBLOCK
        grid_x = ceil_div(xnumel, XBLOCK)
        return (grid_x,)

    else:
        # Unknown type: default to 1D behavior
        grid_x = ceil_div(xnumel, XBLOCK)
        return (grid_x,)


def _infer_kernel_type(numel_dict: Dict[str, int], block_config: Dict[str, int]) -> str:
    """
    Infer kernel type from numel and block configuration.

    Args:
        numel_dict: Dictionary of numel parameters
        block_config: Dictionary of block size parameters

    Returns:
        Kernel type: '1d', '2d', 'reduction', 'reduction_fixed', or 'unknown'
    """
    has_ynumel = 'ynumel' in numel_dict
    has_r0_numel = 'r0_numel' in numel_dict
    has_YBLOCK = 'YBLOCK' in block_config
    has_R0_BLOCK = 'R0_BLOCK' in block_config

    # Reduction kernels
    if has_r0_numel or has_R0_BLOCK:
        if has_R0_BLOCK:
            return 'reduction'
        else:
            return 'reduction_fixed'

    # 2D kernels
    if has_ynumel and has_YBLOCK:
        return '2d'

    # Default to 1D
    return '1d'


def grid_1d(xnumel: int, XBLOCK: int) -> Tuple[int]:
    """
    Compute 1D grid dimensions.

    Args:
        xnumel: Total number of elements
        XBLOCK: Block size in X dimension

    Returns:
        1D grid tuple (grid_x,)
    """
    return (ceil_div(xnumel, XBLOCK),)


def grid_2d(xnumel: int, ynumel: int, XBLOCK: int, YBLOCK: int) -> Tuple[int, int]:
    """
    Compute 2D grid dimensions.

    Args:
        xnumel: Total elements in X dimension
        ynumel: Total elements in Y dimension
        XBLOCK: Block size in X dimension
        YBLOCK: Block size in Y dimension

    Returns:
        2D grid tuple (grid_x, grid_y)
    """
    return (ceil_div(xnumel, XBLOCK), ceil_div(ynumel, YBLOCK))


def infer_numel_for_kernel(src: str, kernel_name: str) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[List[str]]]:
    """
    Find the Triton kernel by name, capture its parameter list (inside the
    parentheses) and its body, then:
      - parse and return the parameter names
      - look for parameters ending with 'numel' and extract their values from body or call sites
      - return separate lists for numel param names, their values, and other param names

    Returns: (numel_param_names, numel_values, other_param_names)
    """
    # Capture both the params and the body with named groups
    pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\((?P<params>.*?)\)\s*:(?P<body>.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m:
        return None, None, None

    params_src = m.group("params").strip()
    body = m.group("body")

    # Robustly parse parameter *names* via AST (handles annotations/defaults)
    all_param_names: Optional[List[str]] = None
    try:
        dummy = f"def __tmp__({params_src}):\n    pass\n"
        mod = ast.parse(dummy)
        f = mod.body[0]  # type: ignore[index]
        # Ensure f is a FunctionDef before accessing args
        if isinstance(f, ast.FunctionDef):
            # Collect positionals; add others if you care (vararg/kwonly/kwarg)
            posonly = getattr(f.args, "posonlyargs", [])
            args = f.args.args
            all_param_names = [a.arg for a in (*posonly, *args)]
    except Exception:
        # Fallback: naive split (good enough for simple signatures)
        if params_src:
            all_param_names = [p.split(":", 1)[0].split("=", 1)[0].strip()
                              for p in params_src.split(",") if p.strip()]

    if not all_param_names:
        return None, None, None

    # Separate numel parameters from other parameters
    numel_param_names = [name for name in all_param_names if name.endswith("numel")]
    other_param_names = [name for name in all_param_names if not name.endswith("numel")]

    # Extract values for all numel parameters
    numel_values = []

    for numel_param in numel_param_names:
        # First try to find literal assignment in body
        pattern = rf"\b{re.escape(numel_param)}\s*=\s*(\d+)"
        m_literal = re.search(pattern, body)

        if m_literal:
            value = int(m_literal.group(1))
            # log.debug(f"Found {numel_param}={value} in kernel body")
            numel_values.append(value)
        else:
            # Try to extract from call sites
            value = _extract_numel_from_call_sites(src, kernel_name, numel_param, all_param_names)

            if value is None:
                # Try to infer from other patterns in the body if this is xnumel
                if numel_param == "xnumel":
                    other_numel = re.search(r"\b(y|z|r0_)?numel\s*=\s*(\d+)", body)
                    if other_numel:
                        value = int(other_numel.group(2))
                        # log.debug(f"Found {other_numel.group(1) or ''}numel={value}, using this as {numel_param}")
                    else:
                        value = 16384  # Fallback for xnumel
                        # log.debug(f"Using fallback {numel_param}={value} (common Inductor size)")
                else:
                    # For other numel parameters, use reasonable defaults
                    if numel_param == "ynumel":
                        value = 16384  # Default for 2D kernels
                    elif numel_param.startswith("r") and numel_param.endswith("_numel"):
                        value = 64  # Default for reduction dimensions
                    else:
                        value = 1024  # General fallback
                    # log.debug(f"Using fallback {numel_param}={value}")

            numel_values.append(value)

    # log.debug(f"Extracted numel parameters: {dict(zip(numel_param_names, numel_values))}")
    # log.debug(f"Other parameters: {other_param_names}")

    return numel_param_names, numel_values, other_param_names

def _extract_numel_from_call_sites(src: str, kernel_name: str, numel_param: str, param_names: List[str]) -> Optional[int]:
    """
    Scan the source code for kernel invocation sites to extract actual numel values.
    Looks for patterns like: kernel_name[grid](*args, numel_value, ...)
    """
    # Find kernel invocation patterns
    # Pattern: kernel_name[grid_size, ...](arg1, arg2, numel_value, ...)
    call_pattern = rf"{re.escape(kernel_name)}\[[^\]]+\]\s*\([^)]+\)"

    call_matches = re.finditer(call_pattern, src)

    for call_match in call_matches:
        call_text = call_match.group(0)
        # log.debug(f"Found kernel call: {call_text[:100]}...")

        # Extract arguments from the call
        # Find the argument list inside parentheses
        args_match = re.search(r'\([^)]+\)', call_text)
        if not args_match:
            continue

        args_text = args_match.group(0)[1:-1]  # Remove parentheses

        # Split arguments, being careful about nested calls
        args = _split_function_args(args_text)

        # Try to find numel parameter based on parameter position
        try:
            if numel_param in param_names:
                numel_idx = param_names.index(numel_param)
                if numel_idx < len(args):
                    arg_value = args[numel_idx].strip()
                    # Try to extract numeric value
                    if arg_value.isdigit():
                        extracted_numel = int(arg_value)
                        # log.debug(f"Extracted {numel_param}={extracted_numel} from call site")
                        return extracted_numel
                    else:
                        pass
                        # log.debug(f"Found {numel_param} argument '{arg_value}' but it's not a simple integer")
        except (ValueError, IndexError):
            continue

    return None


def _split_function_args(args_text: str) -> List[str]:
    """
    Split function arguments, handling nested parentheses and brackets.
    """
    args = []
    current_arg = ""
    paren_depth = 0
    bracket_depth = 0

    for char in args_text:
        if char == ',' and paren_depth == 0 and bracket_depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
            current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())

    return args