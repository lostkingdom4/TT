import re, ast, io, os, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

# ---------------------------
# Parse assert_size_stride(...)
# ---------------------------
def _parse_tuple_ints(txt: str):
    """
    Robustly parse the inner text of a Python tuple of ints.
    Works for '(4,)', '(4, 4, 3, 3)', and even '4' (just in case).
    """
    # The regex captured only the inside; wrap it back in parentheses
    obj = ast.literal_eval(f"({txt})")
    if isinstance(obj, int):
        return (int(obj),)
    # Ensure it's a tuple/list of ints
    return tuple(int(x) for x in obj)

_ASSERT_RE = re.compile(
    r"assert_size_stride\(\s*([A-Za-z0-9_]+)\s*,\s*\(([^)]*)\)\s*,\s*\(([^)]*)\)\s*\)"
)

def parse_asserts(src: str):
    """
    Returns: { var_name: (shape_tuple, stride_tuple) }
    """
    out = {}
    for m in _ASSERT_RE.finditer(src):
        var = m.group(1)
        shape = _parse_tuple_ints(m.group(2))
        stride = _parse_tuple_ints(m.group(3))
        out[var] = (shape, stride)
    return out

# ---------------------------
# Make tensor with given stride
# ---------------------------
def make_tensor_with_stride(
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a tensor with the exact logical shape & stride.
    Strides are in elements (not bytes). Assumes non-negative strides and offset=0.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(shape) != len(stride):
        raise ValueError(f"rank mismatch: shape={shape}, stride={stride}")

    # Edge cases
    if any(d == 0 for d in shape):
        # Zero-size tensor: no elements needed
        base = torch.empty(0, dtype=dtype, device=device)
        return base.as_strided(size=shape, stride=stride)

    # Correct bound: sum, not max
    # highest index touched (offset=0) is sum_i (size_i-1)*stride_i
    upper_index = sum((int(s) - 1) * int(st) for s, st in zip(shape, stride))
    storage_elems = int(upper_index) + 1  # elements, not bytes

    base = torch.empty(storage_elems, dtype=dtype, device=device)
    # as_strided view over the storage; clone to get a dense, writable tensor with same strides
    t_view = base.as_strided(size=tuple(map(int, shape)),
                             stride=tuple(map(int, stride)))
    t = t_view.clone()  # keep the same logical layout but own storage
    t.uniform_()
    return t

# ---------------------------
# Triton kernel parsing
# ---------------------------
_KFUN_RE = re.compile(
    r"@triton\.jit\s*def\s+(triton_poi_[A-Za-z0-9_]+)\s*\((?P<params>.*?)\)\s*:(?P<body>.+?)(?=\n@triton\.jit|\ndef\s|\Z)",
    flags=re.DOTALL
)

def parse_triton_kernels(src: str) -> List[Tuple[str, List[str], Optional[int]]]:
    """
    Finds each @triton.jit kernel, returns list of:
      (kernel_name, param_names, xnumel_literal_or_None)
    """
    out = []
    for m in _KFUN_RE.finditer(src):
        kname = m.group(1)
        params_src = m.group("params").strip()
        body = m.group("body")

        # Get parameter names robustly
        param_names: List[str] = []
        try:
            dummy = f"def __tmp__({params_src}):\n    pass\n"
            mod = ast.parse(dummy)
            f = mod.body[0]
            posonly = getattr(f.args, "posonlyargs", [])
            args = f.args.args
            param_names = [a.arg for a in (*posonly, *args)]
        except Exception:
            if params_src:
                param_names = [p.split(":",1)[0].split("=",1)[0].strip()
                               for p in params_src.split(",") if p.strip()]

        xm = re.search(r"\bxnumel\s*=\s*(\d+)", body)
        xnumel = int(xm.group(1)) if xm else None
        out.append((kname, param_names, xnumel))
    return out

# ---------------------------
# Generate kernel-flat inputs
# ---------------------------
def make_kernel_args(param_names: List[str], xnumel: int, device: torch.device):
    """
    Heuristic: for all params before 'xnumel', allocate a flat buffer of length xnumel.
    This is safe even if a kernel only uses a subset (e.g., bias vector length 4).
    Returns a list of tensors and the index of xnumel within params (for reference).
    """
    args = []
    xidx = None
    for i, n in enumerate(param_names):
        if n == "xnumel":
            xidx = i
            break
    if xidx is None:
        # no xnumel param present; assume all up to first constexpr are pointers
        xidx = len(param_names)

    # allocate float32 buffers for pointer-like args
    for _ in range(xidx):
        args.append(torch.empty(xnumel, dtype=torch.float32, device=device).uniform_())
    # append the integer xnumel at call time (not as a tensor here)
    return args, xidx

# ---------------------------
# Main entry
# ---------------------------
def fabricate_inputs_for_file(py_file: str, out_dir: str) -> Dict:
    """
    - Parses all assert_size_stride -> creates primals_* tensors (and any others asserted)
    - Parses every Triton kernel -> creates flat buffers (length xnumel) per pointer arg
    - Saves:
        {out_dir}/assert_tensors.pt        # dict of asserted tensors
        {out_dir}/{kernel_name}_args.pt    # list of tensors (pre-xnumel)
        {out_dir}/{kernel_name}_meta.json  # {"xnumel": ..., "param_names": [...]}
    Returns a small manifest dict.
    """
    src = Path(py_file).read_text(encoding="utf-8", errors="ignore")
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Assert tensors
    asserts = parse_asserts(src)
    tensors: Dict[str, torch.Tensor] = {}
    for name, (shape, stride) in asserts.items():
        tensors[name] = make_tensor_with_stride(shape, stride, device=device)
    torch.save(tensors, outp / "assert_tensors.pt")

    # 2) Triton kernels + flat args
    kernels = parse_triton_kernels(src)
    manifest = {"file": py_file, "device": str(device), "kernels": []}

    for kname, params, xnumel in kernels:
        if xnumel is None:
            # fallback: if cannot infer, try to deduce from first assert of a 4D activation (e.g., (N,C,H,W))
            # else skip this kernel
            xnumel_guess = None
            for (shape, _stride) in asserts.values():
                if len(shape) >= 1:
                    prod = 1
                    for d in shape:
                        prod *= d
                    if prod > 0:
                        xnumel_guess = prod
                        break
            xnumel = xnumel_guess
        if xnumel is None:
            # nothing to do
            continue

        kargs, xidx = make_kernel_args(params, xnumel, device)
        torch.save(kargs, outp / f"{kname}_args.pt")
        meta = {"kernel": kname, "xnumel": xnumel, "param_names": params, "xnumel_param_index": xidx}
        (outp / f"{kname}_meta.json").write_text(json.dumps(meta, indent=2))
        manifest["kernels"].append(meta)

    # 3) Save a manifest
    (outp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

# ---------------------------
# Convenience: run this file
# ---------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python fabricate_inputs.py <path/to/file.py> <out_dir>")
        raise SystemExit(2)
    mf = fabricate_inputs_for_file(sys.argv[1], sys.argv[2])
    print(json.dumps(mf, indent=2))