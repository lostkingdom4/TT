# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import functools
import logging
import os
import platform
import random
import resource
import signal
import subprocess
import sys
import tempfile
import time


import numpy as np
import torch
import torch.utils._pytree as pytree

from paritybench.reporting import ErrorAggregatorDict, Stats
from run_and_check import import_Model_and_args_from_code
from torch import multiprocessing
from torch._dynamo.utils import clone_inputs
from torch._inductor.utils import fresh_inductor_cache

log = logging.getLogger(__name__)


def call_with_timeout(fn, args, kwargs=None, timeout=10):
	kwargs = kwargs or {}
	parent_conn, child_conn = multiprocessing.Pipe()
	start = time.time()
	proc = multiprocessing.Process(
		target=call_with_timeout_subproc, args=(fn, args, kwargs, child_conn)
	)
	proc.start()

	try:
		while proc.is_alive():
			if parent_conn.poll(1):
				result = parent_conn.recv()
				proc.join()
				return result
			if time.time() - start > timeout:
				# Send SIGINT first to allow for graceful shutdown and stack trace
				os.kill(proc.pid, signal.SIGINT)
				# Give the process a short time to cleanup
				proc.join(1)
				if proc.is_alive():
					# Force terminate if still running
					proc.terminate()
					proc.join(1)
					if proc.is_alive():
						# Kill as last resort
						proc.kill()
						proc.join()
				raise TimeoutError(f'took longer than {timeout} seconds')
	finally:
		# Ensure resources are cleaned up
		if proc.is_alive():
			proc.kill()
			proc.join()

	if proc.exitcode == 0:
		return parent_conn.recv()
	else:
		raise OSError(f'exitcode should be 0, got {proc.exitcode}')


def call_with_timeout_subproc(fn, args, kwargs, return_pipe):
	use_rlimit = (
		os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 1024**3 < 1000
		if platform.system() == 'Linux'
		else True
	)
	if use_rlimit:
		_, hard = resource.getrlimit(resource.RLIMIT_AS)
		resource.setrlimit(
			resource.RLIMIT_AS,
			(int(os.environ.get('RLIMIT_AS_GB', 10)) * 1024**3, hard),
		)
	try:
		result = fn(*args, *kwargs)
		return_pipe.send(result)
	except Exception:
		log.exception('Error from subprocess')
		sys.exit(1)


def import_file(path):
	"""
	:param path: to a *.py file
	:return: a python module
	"""
	code = open(path).read()
	module_name = path.split('/')[-1].split('.')[1]
	model, get_inputs, get_init_inputs = import_Model_and_args_from_code(code, module_name)
	return model, get_inputs, get_init_inputs


# todo: modify call with timeout to something longer but does not crash
def subproc_wrapper(path: str, fn: callable, timeout: int = 300, fresh_cache_dir: bool = False):
	"""
	A wrapper around call_with_timeout() adding a temp dir and error handling.

	:param path: path to code to test
	:param fn: function to run in subprocess
	:param timeout: seconds to wait
	:return: errors, stats
	"""
	log.info(f'Running {path}')
	with (
		tempfile.TemporaryDirectory(prefix='paritybench') as tempdir,
		fresh_inductor_cache() if fresh_cache_dir else contextlib.nullcontext(),
	):
		try:
			return call_with_timeout(fn, (tempdir, path), {}, timeout=timeout)
		except TimeoutError:
			return ErrorAggregatorDict.single(
				'meta', TimeoutError('Timeout testing module'), path
			), Stats({'timeout': 1})
		except OSError:
			return ErrorAggregatorDict.single('meta', OSError('Crash testing module'), path), Stats({
				'crash': 1
			})


def tempdir_wrapper(path: str, fn: callable, fresh_cache_dir: bool = False):
	"""Non-forking version of subproc_wrapper"""
	log.info(f'Running {path}')
	with (
		tempfile.TemporaryDirectory(prefix='paritybench') as tempdir,
		fresh_inductor_cache() if fresh_cache_dir else contextlib.nullcontext(),
	):
		return fn(tempdir, path)


def get_best_gpu_device():
	"""
	Find the GPU with the most free memory.
	Returns the device with the most available memory.
	"""
	if not torch.cuda.is_available():
		return torch.device('cpu')

	best_device = 0
	max_free_memory = 0

	for i in range(torch.cuda.device_count()):
		total_memory = torch.cuda.get_device_properties(i).total_memory
		allocated = torch.cuda.memory_allocated(i)
		free_memory = total_memory - allocated

		log.info(f"GPU {i}: {free_memory / 1e9:.1f} GB free out of {total_memory / 1e9:.1f} GB")

		if free_memory > max_free_memory:
			max_free_memory = free_memory
			best_device = i

	selected_device = torch.device(f'cuda:{best_device}')
	log.info(f"Selected {selected_device} with {max_free_memory / 1e9:.1f} GB free memory")
	return selected_device


def wrap_args(args, device='cuda'):
	# Auto-select best GPU if using CUDA
	if device == 'cuda' and torch.cuda.is_available():
		device = get_best_gpu_device()
	else:
		device = torch.device(device)

	def move_tensor_with_memory_info(tensor, target_device):
		try:
			# Try to clear cache before allocation
			if torch.cuda.is_available() and target_device.type == 'cuda':
				torch.cuda.empty_cache()
			return tensor.to(target_device)
		except RuntimeError as e:
			if "CUDA out of memory" in str(e) or "out of memory" in str(e):
				# Get detailed memory information
				if torch.cuda.is_available() and target_device.type == 'cuda':
					gpu_id = target_device.index or 0

					# Clear cache and get fresh memory stats
					torch.cuda.empty_cache()
					allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
					reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
					max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9
					total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

					tensor_size = tensor.numel() * tensor.element_size() / 1e9

					# Additional system info
					import subprocess
					try:
						nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
						gpu_memory_info = nvidia_smi_output.strip().split('\n')[gpu_id] if gpu_id < len(nvidia_smi_output.strip().split('\n')) else "N/A"
					except:
						gpu_memory_info = "nvidia-smi unavailable"

					# Check for memory fragmentation
					try:
						# Try allocating a small test tensor
						test_tensor = torch.zeros(1, device=target_device)
						test_success = "SUCCESS"
						del test_tensor
					except:
						test_success = "FAILED"

					error_msg = (
						f"CUDA out of memory when moving tensor to {target_device}:\n"
						f"  Tensor size: {tensor_size:.6f} GB (shape: {list(tensor.shape)}, dtype: {tensor.dtype})\n"
						f"  Tensor source device: {tensor.device}\n"
						f"  GPU {gpu_id} PyTorch memory:\n"
						f"    Allocated: {allocated:.6f} GB\n"
						f"    Reserved: {reserved:.6f} GB\n"
						f"    Max allocated: {max_allocated:.6f} GB\n"
						f"    Total GPU memory: {total_memory:.6f} GB\n"
						f"    Free memory: {total_memory - reserved:.6f} GB\n"
						f"  GPU {gpu_id} system memory (nvidia-smi): {gpu_memory_info}\n"
						f"  Small tensor allocation test: {test_success}\n"
						f"  CUDA context initialized: {torch.cuda.is_initialized()}\n"
						f"  PyTorch CUDA version: {torch.version.cuda}\n"
						f"  Original error: {str(e)}\n"
						f"  \n"
						f"  DIAGNOSIS: This appears to be a CUDA driver/system issue, not actual memory shortage.\n"
						f"  Try: 1) Restart the script 2) nvidia-smi --gpu-reset-gpc 3) Check CUDA driver"
					)
					raise RuntimeError(error_msg) from e
			raise e

	return [move_tensor_with_memory_info(x, device) if isinstance(x, torch.Tensor) else x
	        for x in copy.deepcopy(args)]


def wrap_kwargs(kwargs, device='cuda'):
	# Auto-select best GPU if using CUDA
	if device == 'cuda' and torch.cuda.is_available():
		device = get_best_gpu_device()
	else:
		device = torch.device(device)

	def move_tensor_with_memory_info(tensor, target_device):
		try:
			return tensor.clone().to(target_device)
		except RuntimeError as e:
			if "CUDA out of memory" in str(e) or "out of memory" in str(e):
				# Get detailed memory information
				if torch.cuda.is_available() and target_device.type == 'cuda':
					gpu_id = target_device.index or 0
					allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
					reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
					max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9
					total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

					tensor_size = tensor.numel() * tensor.element_size() / 1e9

					error_msg = (
						f"CUDA out of memory when moving tensor to {target_device}:\n"
						f"  Tensor size: {tensor_size:.3f} GB (shape: {list(tensor.shape)}, dtype: {tensor.dtype})\n"
						f"  GPU {gpu_id} memory usage:\n"
						f"    Allocated: {allocated:.3f} GB\n"
						f"    Reserved: {reserved:.3f} GB\n"
						f"    Max allocated: {max_allocated:.3f} GB\n"
						f"    Total GPU memory: {total_memory:.3f} GB\n"
						f"    Free memory: {total_memory - reserved:.3f} GB\n"
						f"  Original error: {str(e)}"
					)
					raise RuntimeError(error_msg) from e
			raise e

	wrapped_kwargs = {}
	for k, v in kwargs.items():
		if isinstance(v, torch.Tensor):
			wrapped_kwargs.update({k: move_tensor_with_memory_info(v, device)})
		else:
			wrapped_kwargs.update({k: copy.deepcopy(v)})
	return wrapped_kwargs


def get_skiplist(main_args):
	if main_args.compile_mode == 'export':
		return SKIP.get('export')
	else:
		return SKIP.get(main_args.backend)


def get_tol(main_args):
	if main_args.backend == 'inductor':
		return INDUCTOR_TOL
	else:
		return DYNAMO_TOL


@functools.lru_cache(None)
def patch_torch_manual_seed():
	"""Make torch manual seed deterministic. Helps with accuracy testing."""

	def deterministic_torch_manual_seed(*args, **kwargs):
		from torch._C import default_generator

		seed = 1337
		import torch.cuda

		if not torch.cuda._is_in_bad_fork():
			torch.cuda.manual_seed_all(seed)
		return default_generator.manual_seed(seed)

	torch.manual_seed = deterministic_torch_manual_seed


def reset_rng_state():
	torch.manual_seed(1337)
	random.seed(1337)
	np.random.seed(1337)


def cast_to(dtype, model, inputs):
	# cast model and inputs to fp16
	if dtype == torch.float16:
		model = model.half()
	else:
		model = model.to(dtype)

	inputs = pytree.tree_map(
		lambda x: (x.to(dtype) if isinstance(x, torch.Tensor) and x.is_floating_point() else x),
		inputs,
	)
	return model, inputs


def cast_to_fp64(model, inputs):
	return cast_to(torch.float64, model, inputs)


def get_cosine_and_fp64_outputs(model, args, kwargs):
	# Collect the fp64 reference outputs to be used later for accuracy checking.
	fp64_outputs = None
	cosine = False
	reset_rng_state()
	try:
		model_fp64, args_fp64, kwargs_fp64 = cast_to_fp64(
			copy.deepcopy(model),
			clone_inputs(args),
			clone_inputs(kwargs),
		)
		fp64_outputs = model_fp64(*args_fp64, **kwargs_fp64)
	except Exception:
		log.warning(
			'fp64 golden ref were not generated. Setting accuracy check to cosine',
		)
		cosine = True
		fp64_outputs = None
	return cosine, fp64_outputs


DYNAMO_TOL = 1e-4
INDUCTOR_TOL = 1e-3

SKIP_DYNAMO_EAGER = [
	'./generated/test_deepinsight_insightface.py:deeplab_xception_transfer_basemodel',
	'./generated/test_SforAiDl_vformer.py:PVTClassification',
	'./generated/test_SforAiDl_vformer.py:PVTDetection',  # try ... catch ...
	'./generated/test_BlinkDL_RWKV_LM.py:RWKV_ChannelMix',  # Subclasses torch.jit.ScriptModule
	'./generated/test_facebookresearch_pytext.py:ContextualTokenEmbedding',
	'./generated/test_pytorch_translate.py:OutputProjection',  # shape_as_tensor
	'./generated/test_ludwig_ai_ludwig.py:_VectorPreprocessing',
	'./generated/test_ludwig_ai_ludwig.py:_DatePreprocessing',  # torch.jit.isinstance
	'./generated/test_adapter_hub_adapter_transformers.py:GPTNeoSelfAttention',  # torch.where with dtype torch.uint8 is now deprecated.
	'./generated/test_ZhaoJ9014_face_evoLVe.py:AM_Softmax',
	'./generated/test_ZhaoJ9014_face_evoLVe.py:CircleLoss',
	'./generated/test_ZhaoJ9014_face_evoLVe.py:MagFace',
	'./generated/test_fangchangma_self_supervised_depth_completion.py:PhotometricLoss',  # torch.index with dtype torch.uint8 is now deprecated.
	'./generated/test_megvii_research_NAFNet.py:PSNRLoss',
	'./generated/test_murufeng_FUIR.py:PSNRLoss',
	'./generated/test_swz30_Restormer.py:PSNRLoss',  # Dynamo convert numpy.float64 to python float32.
]
SKIP_INDUCTOR = [
	'./generated/test_BIGBALLON_CIFAR_ZOO.py:SKConv',
	'./generated/test_CDOTAD_AlphaGAN_Matting.py:_AsppBlock',
	'./generated/test_CDOTAD_AlphaGAN_Matting.py:Encoder',
	'./generated/test_LikeLy_Journey_SegmenTron.py:_GlobalAvgPooling',
	'./generated/test_Minerva_J_Pytorch_Segmentation_multi_models.py:ChainedResidualPoolImproved',
	'./generated/test_NVIDIA_CleanUNet.py:CleanUNet',
	'./generated/test_VITA_Group_DeblurGANv2.py:ConvBlock',
	'./generated/test_Tramac_awesome_semantic_segmentation_pytorch.py:_GlobalAvgPooling',
	'./generated/test_Tramac_awesome_semantic_segmentation_pytorch.py:XceptionA',
	'./generated/test_alibaba_EasyCV.py:ClassAttentionBlock',
	'./generated/test_arnab39_cycleGAN_PyTorch.py:ResidualBlock',
	'./generated/test_chengchunhsu_EveryPixelMatters.py:Bottleneck',
	'./generated/test_deepinsight_insightface.py:DeepLabv3_plus',
	'./generated/test_deepinsight_insightface.py:Xception',
	'./generated/test_egorzakharov_PerceptualGAN.py:ResBlock',
	'./generated/test_fidler_lab_defgrid_release.py:PPM',
	'./generated/test_iscyy_yoloair.py:BottleneckCSPDNL',
	'./generated/test_iscyy_yoloair.py:DNL',
	'./generated/test_jfzhang95_pytorch_deeplab_xception.py:InvertedResidual',
	'./generated/test_lescientifik_open_brats2020.py:UBlock',
	'./generated/test_ltkong218_IFRNet.py:ResBlock',
	'./generated/test_nihalsid_ViewAL.py:InvertedResidual',
	'./generated/test_sithu31296_semantic_segmentation.py:PSAP',
	'./generated/test_voldemortX_pytorch_auto_drive.py:SpatialConv',
	'./generated/test_yassouali_pytorch_segmentation.py:_PSPModule',  # Backward fails.
]
SKIP = {
	'eager': SKIP_DYNAMO_EAGER,
	'inductor': SKIP_DYNAMO_EAGER + SKIP_INDUCTOR,
	'export': SKIP_DYNAMO_EAGER,
}


def reset_cuda_context():
	"""
	Reset CUDA context to fix driver-level memory issues.
	Call this when getting spurious CUDA OOM errors with plenty of free memory.
	"""
	try:
		if torch.cuda.is_available():
			# Clear all cached memory
			torch.cuda.empty_cache()
			# Reset memory stats
			torch.cuda.reset_peak_memory_stats()
			torch.cuda.reset_accumulated_memory_stats()
			log.info("CUDA context reset successfully")
			return True
	except Exception as e:
		log.warning(f"Failed to reset CUDA context: {e}")
		return False
