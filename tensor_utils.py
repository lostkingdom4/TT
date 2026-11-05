#!/usr/bin/env python3
"""
Utility functions for working with captured tensor shapes.

Provides functions to:
1. Load tensor shape metadata from JSON files
2. Create random tensors matching captured shapes
3. Recreate kernel inputs for testing/benchmarking
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def load_tensor_shapes(shapes_file: Path) -> Dict[str, Any]:
    """
    Load tensor shape metadata from a JSON file.

    Args:
        shapes_file: Path to *_shapes.json file

    Returns:
        Dictionary with tensor metadata including shapes, dtypes, grid, etc.
    """
    with open(shapes_file, 'r') as f:
        return json.load(f)


def dtype_from_string(dtype_str: str) -> torch.dtype:
    """
    Convert string representation of dtype to torch.dtype.

    Args:
        dtype_str: String like 'torch.float32', 'torch.int64', etc.

    Returns:
        Corresponding torch.dtype
    """
    dtype_map = {
        'torch.float32': torch.float32,
        'torch.float': torch.float32,
        'torch.float64': torch.float64,
        'torch.double': torch.float64,
        'torch.float16': torch.float16,
        'torch.half': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.int32': torch.int32,
        'torch.int': torch.int32,
        'torch.int64': torch.int64,
        'torch.long': torch.int64,
        'torch.int16': torch.int16,
        'torch.short': torch.int16,
        'torch.int8': torch.int8,
        'torch.uint8': torch.uint8,
        'torch.bool': torch.bool,
    }

    return dtype_map.get(dtype_str, torch.float32)


def create_random_tensors(shapes_file: Path,
                          device: str = 'cuda',
                          seed: Optional[int] = None) -> Tuple[List[torch.Tensor], List[Any]]:
    """
    Create random tensors matching the shapes from a captured kernel.

    Args:
        shapes_file: Path to *_shapes.json file
        device: Device to create tensors on ('cuda', 'cpu', 'xpu', etc.)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (tensor_list, scalar_list) matching the kernel's expected inputs
    """
    if seed is not None:
        torch.manual_seed(seed)

    metadata = load_tensor_shapes(shapes_file)

    # Create tensors from tensor_args
    tensors = []
    tensor_args = metadata.get('tensor_args', [])

    for tensor_info in tensor_args:
        shape = tuple(tensor_info['shape'])
        dtype = dtype_from_string(tensor_info['dtype'])

        # Create random tensor with the specified shape and dtype
        if dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            tensor = torch.randn(shape, dtype=dtype, device=device)
        elif dtype == torch.bool:
            tensor = torch.randint(0, 2, shape, dtype=dtype, device=device)
        else:  # integer types
            tensor = torch.randint(-100, 100, shape, dtype=dtype, device=device)

        tensors.append(tensor)

    # Extract scalar arguments
    scalars = []
    scalar_args = metadata.get('scalar_args', [])

    for scalar_info in scalar_args:
        scalars.append(scalar_info['value'])

    return tensors, scalars


def create_kernel_inputs(shapes_file: Path,
                        device: str = 'cuda',
                        values: str = 'random',
                        seed: Optional[int] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Create complete kernel input list (tensors + scalars) and kwargs in correct order.

    Args:
        shapes_file: Path to *_shapes.json file
        device: Device to create tensors on
        values: 'random', 'zeros', 'ones', or 'randn'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (args_list, kwargs_dict) for kernel invocation
    """
    if seed is not None:
        torch.manual_seed(seed)

    metadata = load_tensor_shapes(shapes_file)

    # Combine tensor and scalar args in correct order
    tensor_args = metadata.get('tensor_args', [])
    scalar_args = metadata.get('scalar_args', [])
    total_args = metadata.get('total_args', len(tensor_args) + len(scalar_args))

    # Create mapping of arg_index to value
    args_dict = {}

    # Process tensor arguments
    for tensor_info in tensor_args:
        idx = tensor_info['arg_index']
        shape = tuple(tensor_info['shape'])
        dtype = dtype_from_string(tensor_info['dtype'])

        # Create tensor based on values parameter
        if values == 'zeros':
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        elif values == 'ones':
            tensor = torch.ones(shape, dtype=dtype, device=device)
        elif values == 'randn' or values == 'random':
            if dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                tensor = torch.randn(shape, dtype=dtype, device=device)
            elif dtype == torch.bool:
                tensor = torch.randint(0, 2, shape, dtype=dtype, device=device)
            else:
                tensor = torch.randint(-100, 100, shape, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown values parameter: {values}")

        args_dict[idx] = tensor

    # Process scalar arguments
    for scalar_info in scalar_args:
        idx = scalar_info['arg_index']
        args_dict[idx] = scalar_info['value']

    # Build ordered argument list
    args = []
    for i in range(total_args):
        if i in args_dict:
            args.append(args_dict[i])
        else:
            # Argument not captured - should not happen
            raise ValueError(f"Missing argument at index {i}")

    # Process kwargs
    kwargs = {}
    kwargs_info = metadata.get('kwargs', {})

    for key, value_info in kwargs_info.items():
        if value_info['type'] == 'tensor':
            shape = tuple(value_info['shape'])
            dtype = dtype_from_string(value_info['dtype'])
            if values == 'zeros':
                kwargs[key] = torch.zeros(shape, dtype=dtype, device=device)
            elif values == 'ones':
                kwargs[key] = torch.ones(shape, dtype=dtype, device=device)
            else:
                if dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
                    kwargs[key] = torch.randn(shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    kwargs[key] = torch.randint(0, 2, shape, dtype=dtype, device=device)
                else:
                    kwargs[key] = torch.randint(-100, 100, shape, dtype=dtype, device=device)
        elif value_info['type'] == 'scalar':
            kwargs[key] = value_info['value']
        else:
            # Skip 'other' types as we can't reconstruct them
            pass

    return args, kwargs


def get_tensor_summary(shapes_file: Path) -> Dict[str, Any]:
    """
    Get a summary of tensor shapes and metadata for a kernel.

    Args:
        shapes_file: Path to *_shapes.json file

    Returns:
        Summary dictionary with statistics
    """
    metadata = load_tensor_shapes(shapes_file)

    tensor_args = metadata.get('tensor_args', [])
    scalar_args = metadata.get('scalar_args', [])

    total_memory = sum(
        tensor_info['numel'] * torch.finfo(dtype_from_string(tensor_info['dtype'])).bits // 8
        if dtype_from_string(tensor_info['dtype']).is_floating_point
        else tensor_info['numel'] * 4  # Assume 4 bytes for integers
        for tensor_info in tensor_args
    )

    return {
        'kernel_name': metadata.get('kernel_name', 'unknown'),
        'module_file': metadata.get('module_file', 'unknown'),
        'num_tensor_args': len(tensor_args),
        'num_scalar_args': len(scalar_args),
        'total_args': metadata.get('total_args', 0),
        'grid': metadata.get('grid', []),
        'tensor_shapes': [t['shape'] for t in tensor_args],
        'tensor_dtypes': [t['dtype'] for t in tensor_args],
        'total_elements': sum(t['numel'] for t in tensor_args),
        'estimated_memory_bytes': total_memory,
        'estimated_memory_mb': total_memory / (1024 * 1024),
    }


def print_tensor_info(shapes_file: Path):
    """
    Print human-readable tensor information.

    Args:
        shapes_file: Path to *_shapes.json file
    """
    summary = get_tensor_summary(shapes_file)

    print(f"Kernel: {summary['kernel_name']}")
    print(f"Module: {summary['module_file']}")
    print(f"Grid: {summary['grid']}")
    print(f"\nArguments:")
    print(f"  Tensors: {summary['num_tensor_args']}")
    print(f"  Scalars: {summary['num_scalar_args']}")
    print(f"  Total: {summary['total_args']}")
    print(f"\nTensor Shapes:")
    for i, (shape, dtype) in enumerate(zip(summary['tensor_shapes'], summary['tensor_dtypes'])):
        print(f"  [{i}] {shape} ({dtype})")
    print(f"\nMemory:")
    print(f"  Total elements: {summary['total_elements']:,}")
    print(f"  Estimated memory: {summary['estimated_memory_mb']:.2f} MB")


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tensor_utils.py <shapes_file.json>")
        sys.exit(1)

    shapes_file = Path(sys.argv[1])

    if not shapes_file.exists():
        print(f"Error: File not found: {shapes_file}")
        sys.exit(1)

    print("="*80)
    print("TENSOR INFORMATION")
    print("="*80)
    print_tensor_info(shapes_file)

    print("\n" + "="*80)
    print("CREATING TEST TENSORS")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    tensors, scalars = create_random_tensors(shapes_file, device=device, seed=42)

    print(f"\nCreated {len(tensors)} tensors:")
    for i, t in enumerate(tensors):
        print(f"  [{i}] shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")

    print(f"\nScalar arguments ({len(scalars)}):")
    for i, s in enumerate(scalars):
        print(f"  [{i}] {s} (type: {type(s).__name__})")

    print("\n" + "="*80)
    print("CREATING COMPLETE KERNEL INPUTS")
    print("="*80)

    all_args, all_kwargs = create_kernel_inputs(shapes_file, device=device, values='random', seed=42)
    print(f"Total arguments: {len(all_args)}")
    for i, arg in enumerate(all_args):
        if isinstance(arg, torch.Tensor):
            print(f"  [{i}] Tensor: shape={tuple(arg.shape)}, dtype={arg.dtype}")
        else:
            print(f"  [{i}] Scalar: {arg} ({type(arg).__name__})")

    if all_kwargs:
        print(f"\nKeyword arguments ({len(all_kwargs)}):")
        for key, value in all_kwargs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape={tuple(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value} ({type(value).__name__})")
