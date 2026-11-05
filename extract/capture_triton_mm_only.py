#!/usr/bin/env python3
"""Capture the `triton_mm` Inductor template per-shape without modifying
any tritonbench or extract/ source.

Context
-------
For the 42 fp16 addmm shapes on V100, Inductor's max-autotune always picks
`triton_tem_fused_addmm_0` (the fused-bias template) over `triton_mm` (the
plain matmul template). Our standard extractor (``extract/run_inductor.py``)
hooks `CachingAutotuner.run`, which only fires for the *winning* template,
so the `triton_mm` candidate never lands on disk as a per-shape snapshot.

This script takes a different hook point: we monkey-patch
``TritonTemplateCaller.precompile`` at runtime. Inductor precompiles *every*
candidate (winning or losing) before benchmarking them, so the patch sees
both `triton_mm` and `triton_tem_fused_addmm_0` candidates. We copy the
``triton_mm``-named candidate's generated .py into the extraction directory
as ``triton_mm_input{N}.py`` for shape N.

No files under ``tb/`` or ``extract/run_inductor.py`` are modified.

Usage
-----
    source <conda_root>/etc/profile.d/conda.sh
    conda activate tb_multi_agent
    python extract/capture_triton_mm_only.py --gpu v100

Output
------
Writes to ``extract/ir_output/<gpu>/addmm/``:
  * ``triton_mm_input{N}.py``   — per-shape template snapshot (1 per shape)
  * ``triton_mm.py``            — representative base copy (shape 0)

Existing ``addmm_triton_mm_input{N}_shapes.json`` files are reused as-is;
they describe the 4-tensor (bias, A, B, out) signature the template takes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as sa

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _install_first_choice_autotune():
	"""Narrow ``AlgorithmSelectorCache.__call__`` choices to ``[choices[0]]``.

	Mirrors ``extract/run_inductor.py::_install_first_choice_autotune``. Without
	this, Inductor precompiles all 19 mm_configs candidates and the first
	``triton_mm``-named one captured by our hook can be any of them. With this,
	only choices[0] = (BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, num_stages=1,
	num_warps=2) is precompiled per shape — matching the choices[0] semantics
	used by the rest of the post-fix snapshot.
	"""
	original_call = sa.AlgorithmSelectorCache.__call__

	def first_choice_call(self, name, choices, *args, **kwargs):
		if choices:
			choices = [choices[0]]
		return original_call(self, name, choices, *args, **kwargs)

	sa.AlgorithmSelectorCache.__call__ = first_choice_call

	def restore():
		sa.AlgorithmSelectorCache.__call__ = original_call

	return restore


def _load_shape(shapes_json: Path):
	"""Instantiate fp16 CUDA tensors for an addmm shape file.

	Schema: tensor_args = [bias (M,N), mat1 (M,K), mat2 (K,N), out (M,N)].
	We return (bias, mat1, mat2); the out tensor is produced by torch.addmm.
	"""
	with shapes_json.open() as f:
		data = json.load(f)
	tensors = data["tensor_args"]
	bias_shape = tensors[0]["shape"]
	mat1_shape = tensors[1]["shape"]
	mat2_shape = tensors[2]["shape"]
	bias = torch.randn(*bias_shape, dtype=torch.float16, device="cuda")
	mat1 = torch.randn(*mat1_shape, dtype=torch.float16, device="cuda")
	mat2 = torch.randn(*mat2_shape, dtype=torch.float16, device="cuda")
	return bias, mat1, mat2


def _install_capture_hook(output_dir: Path, input_id: int, capture_state: dict):
	"""Patch TritonTemplateCaller.precompile to save the first ``triton_mm``
	candidate's .py.

	With ``TORCHINDUCTOR_COMPILE_THREADS=1`` (set by the caller), Inductor
	dispatches precompile serially in choices-list order. The first hook fire
	therefore corresponds to ``choices[0]`` = ``mm_configs[0]`` =
	``(BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, num_stages=1, num_warps=2)``,
	which is the choices[0] semantics we want.

	(Note: we deliberately do *not* narrow ``choices`` to a single element via
	``AlgorithmSelectorCache.__call__``. With a single choice, Inductor
	short-circuits at select_algorithm.py:1385-1388, skips the precompile
	pipeline entirely, and codegens through the fused-with-epilogue path that
	renames the kernel to ``triton_tem_fused_addmm_0``. We need the
	multi-candidate path to keep the ``triton_mm`` kernel name; serial
	precompile gives choices[0] deterministically without narrowing.)

	``capture_state`` is written in-place with keys: ``captured`` (bool),
	``dest_path`` (Path), ``kernel_name`` (str), ``src_path`` (str), ``seen``
	(list of (name, path) tuples).
	"""
	original_precompile = sa.TritonTemplateCaller.precompile

	def _patched(self):
		result = original_precompile(self)
		try:
			name = self.bmreq.kernel_name
			src_path = Path(self.bmreq.module_path)
		except AttributeError:
			return result
		capture_state.setdefault("seen", []).append((name, str(src_path)))
		# Capture the first ``triton_mm``-named candidate (= choices[0] under
		# serial precompile); skip the addmm-fused template and other variants.
		if name.startswith("triton_mm") and not capture_state.get("captured"):
			if src_path.exists():
				dest = output_dir / f"triton_mm_input{input_id}.py"
				shutil.copyfile(src_path, dest)
				capture_state["captured"] = True
				capture_state["dest_path"] = dest
				capture_state["kernel_name"] = name
				capture_state["src_path"] = str(src_path)
		return result

	sa.TritonTemplateCaller.precompile = _patched
	return original_precompile


def _restore_hook(original_precompile):
	sa.TritonTemplateCaller.precompile = original_precompile


def _compile_and_capture(bias, mat1, mat2, output_dir: Path, input_id: int):
	# Isolate Triton's compile cache per shape so we can locate the .ttgir/.ttir
	# that correspond exactly to this shape's choices[0] compilation.
	triton_cache = tempfile.mkdtemp(prefix=f"triton_cache_capture_input{input_id}_")
	prior_triton_cache_env = os.environ.get("TRITON_CACHE_DIR")
	prior_compile_threads = os.environ.get("TORCHINDUCTOR_COMPILE_THREADS")
	os.environ["TRITON_CACHE_DIR"] = triton_cache
	# Force serial precompile so the first ``triton_mm`` candidate the hook
	# sees is choices[0] = mm_configs[0]. With parallel precompile (the
	# default), the first hook fire is whichever worker thread finishes first
	# — non-deterministic across configs.
	os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

	capture_state: dict = {"captured": False, "triton_cache": triton_cache}
	original_precompile = _install_capture_hook(output_dir, input_id, capture_state)
	try:
		torch._dynamo.reset()
		with inductor_config.patch(
			max_autotune=True,
			max_autotune_gemm_backends="TRITON",
			autotune_fallback_to_aten=False,
		):
			f = lambda a, x, y: torch.addmm(a, x, y)
			compiled = torch.compile(f, dynamic=False)
			# First call triggers compile + autotune across candidates
			_ = compiled(bias, mat1, mat2)
	finally:
		_restore_hook(original_precompile)
		# Restore previous env vars (may be unset)
		if prior_triton_cache_env is None:
			os.environ.pop("TRITON_CACHE_DIR", None)
		else:
			os.environ["TRITON_CACHE_DIR"] = prior_triton_cache_env
		if prior_compile_threads is None:
			os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
		else:
			os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = prior_compile_threads
	return capture_state


def _write_canonical_trio(triton_cache: str, output_dir: Path,
                          kernel_name: str, op_name: str = "addmm") -> None:
	"""Emit ``<op>_<kernel>.{ttgir,ttir,json}`` matching ``run_inductor.py``'s
	output schema. Uses the most recent ``<kernel>.ttgir`` / ``<kernel>.ttir``
	in ``triton_cache`` (the choices[0] specialization for this shape).

	``loc()`` references in the TTGIR are rewritten to point at the saved .py.
	"""
	cache = Path(triton_cache)
	ttgir_candidates = sorted(cache.rglob(f"{kernel_name}.ttgir"))
	ttir_candidates = sorted(cache.rglob(f"{kernel_name}.ttir"))
	if not ttgir_candidates or not ttir_candidates:
		print(f"    warning: no {kernel_name}.ttgir/.ttir in {cache}")
		return
	ttgir_src = max(ttgir_candidates, key=lambda p: p.stat().st_mtime)
	ttir_src = max(ttir_candidates, key=lambda p: p.stat().st_mtime)

	saved_py = (output_dir / f"{kernel_name}.py").resolve()
	ttgir_content = ttgir_src.read_text()
	ttgir_content_new = re.sub(
		r'loc\("[^"]*\.py":',
		f'loc("{saved_py}":',
		ttgir_content,
	)

	(output_dir / f"{op_name}_{kernel_name}.ttgir").write_text(ttgir_content_new)
	shutil.copy2(str(ttir_src), str(output_dir / f"{op_name}_{kernel_name}.ttir"))
	(output_dir / f"{op_name}_{kernel_name}.json").write_text(json.dumps({
		"name": kernel_name,
		"operator": op_name,
		"is_inductor_generated": True,
		"arch": "",
		"shared": 0,
	}, indent=2))
	print(f"    wrote {op_name}_{kernel_name}.{{ttgir,ttir,json}} "
	      f"from {ttgir_src.parent.name}/")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu", default="v100")
	parser.add_argument("--output-dir", default=None,
	                    help="Override the addmm output dir. Defaults to "
	                         "extract/ir_output/<gpu>/addmm/.")
	parser.add_argument("--only-input", type=int, default=None,
	                    help="Capture just one input_id (debugging).")
	args = parser.parse_args()

	addmm_dir = (Path(args.output_dir) if args.output_dir
	             else _PROJECT_ROOT / "extract" / "ir_output" / args.gpu / "addmm")
	# Prefer addmm_triton_mm_input*_shapes.json when present (legacy: pre-fix
	# run_inductor.py captured both triton_mm and triton_tem_fused_addmm_0
	# kernels, so both shape files existed). Under the choices[0] hook only
	# triton_tem_fused_addmm_0 is captured at runtime, so fall back to its
	# shape files — schema is identical (same bias/mat1/mat2 tensor_args), only
	# the metadata "kernel" field differs and we don't read it.
	shape_files = sorted(addmm_dir.glob("addmm_triton_mm_input*_shapes.json"))
	if not shape_files:
		shape_files = sorted(
			addmm_dir.glob("addmm_triton_tem_fused_addmm_0_input*_shapes.json")
		)
		if shape_files:
			print(f"[capture_triton_mm_only] no addmm_triton_mm_input*_shapes.json; "
			      f"falling back to addmm_triton_tem_fused_addmm_0_input*_shapes.json "
			      f"({len(shape_files)} files)")
	if not shape_files:
		print(f"[capture_triton_mm_only] no shape files at {addmm_dir}",
		      file=sys.stderr)
		sys.exit(2)

	print(f"[capture_triton_mm_only] {len(shape_files)} shapes -> {addmm_dir}")

	captured = 0
	missed: list[int] = []
	for sf in shape_files:
		stem = sf.stem  # e.g. addmm_triton_mm_input12_shapes
		iid = int(stem.split("input")[-1].rsplit("_", 1)[0])
		if args.only_input is not None and iid != args.only_input:
			continue
		print(f"  [input{iid}] {sf.name}: compiling ...", flush=True)
		try:
			bias, mat1, mat2 = _load_shape(sf)
		except Exception as e:
			print(f"    load failed: {e}")
			missed.append(iid)
			continue
		try:
			state = _compile_and_capture(bias, mat1, mat2, addmm_dir, iid)
		except Exception as e:
			print(f"    compile/capture failed: {e}")
			missed.append(iid)
			continue
		if state.get("captured"):
			captured += 1
			print(f"    saved {state['dest_path'].name} "
			      f"(kernel={state['kernel_name']})")
			# Emit a per-input shapes JSON tagged kernel='triton_mm' so
			# verify_run_list / batch_runner can locate this kernel by name.
			# Schema is identical to the fallback (same bias/mat1/mat2
			# tensor_args); only the "kernel" field differs.
			if sf.stem.startswith("addmm_triton_tem_fused_addmm_0_"):
				try:
					with sf.open() as fh:
						meta = json.load(fh)
					meta["kernel"] = "triton_mm"
					out = addmm_dir / f"addmm_triton_mm_input{iid}_shapes.json"
					with out.open("w") as fh:
						json.dump(meta, fh, indent=2)
				except Exception as e:
					print(f"    warning: could not write triton_mm shapes JSON: {e}")
			# For shape 0, also emit the canonical TTGIR/TTIR/JSON trio so the
			# kernel matches the snapshot layout used by run_inductor.py for
			# every other op.
			if iid == 0:
				_write_canonical_trio(
					state["triton_cache"], addmm_dir,
					kernel_name="triton_mm", op_name="addmm",
				)
		else:
			missed.append(iid)
			seen = state.get("seen", [])
			print(f"    triton_mm candidate not found "
			      f"(saw {len(seen)} candidates: "
			      f"{[n for n,_ in seen[:5]]}{' ...' if len(seen)>5 else ''})")
		# Free GPU memory between shapes
		del bias, mat1, mat2
		torch.cuda.empty_cache()

	# Promote shape-0 capture to the base triton_mm.py that runner.log and
	# benchmark_config expect as the fallback source.
	base = addmm_dir / "triton_mm_input0.py"
	if base.exists():
		shutil.copyfile(base, addmm_dir / "triton_mm.py")

	# Emit a default addmm_triton_mm_shapes.json so multiagent_triton_runner's
	# fallback shape lookup (triggered when --all-shapes is not in effect, e.g.
	# under batch_runner --parallel-shapes which sets --shape-file directly)
	# can resolve the kernel before the per-shape override kicks in.
	default_shape = addmm_dir / "addmm_triton_mm_shapes.json"
	first_input = addmm_dir / "addmm_triton_mm_input0_shapes.json"
	if first_input.exists() and not default_shape.exists():
		shutil.copyfile(first_input, default_shape)

	print(f"[capture_triton_mm_only] captured {captured}/{len(shape_files)} "
	      f"(missed: {missed})")


if __name__ == "__main__":
	main()
