# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset, Dataset
import argparse
import logging
import os
import sys
import multiprocessing as mp
from functools import partial
import pandas as pd

# import torch
# import torch._dynamo

from paritybench.crawler import CrawlGitHub
from paritybench.evaluate import evaluate_all, evaluate_pyfile_subproc
from paritybench.generate import generate_all, generate_zipfile_subproc, write_helpers
from paritybench.utils import subproc_wrapper, tempdir_wrapper
from utils import replace_number_in_functions

log = logging.getLogger(__name__)


def main_one_file(fn, path, args):
	if ':' in path and not args.filter:
		path, args.filter = path.split(':', 2)
	assert os.path.isfile(path) or os.path.isdir(path)

	fn = partial(fn, args=args)

	if not args.no_fork:
		wrapper = subproc_wrapper
	else:
		wrapper = tempdir_wrapper

	errors, stats = wrapper(path, fn=fn, fresh_cache_dir=args.fresh_cache_dir)

	errors.print_report()
	log.info(f'Stats: {stats}')
	return

def load_code(ds, run_dir, args):
	for data in ds:
		repo_name = data['repo_name'].replace('/', '_')
		module_name = data['module_name']
		module_dir = os.path.join(run_dir, 'cleaned_pytorch_modules')
		os.makedirs(module_dir, exist_ok=True)
		code_path = os.path.join(module_dir, f"{repo_name}.{module_name}.py")

		if args.reload:
			log.info(f'Reloading {module_name}')
		elif os.path.exists(code_path):
			log.info(f'Skipping {module_name}, already exists')
			continue

		if not os.path.exists(code_path):
			code = data['python_code']
			with open(code_path, 'w') as f:
				f.write(code)
			log.info(f'Wrote {code_path}')

	if args.reload and args.change_inputs_dims:
		log.info(f'Replacing input dimensions with {args.change_inputs_dims}')
		replace_number_in_functions(module_dir, str(args.change_inputs_dims))
	elif args.reload:
		log.info('Skipping dimension replacement (change_inputs_dims not set)')

def load_code_kernel_bench(ds, run_dir, args):
	module_dir = os.path.join(run_dir, 'cleaned_pytorch_modules')
	os.makedirs(module_dir, exist_ok=True)

	for data in ds:
		repo_name = data['problem_id']
		module_name = data['name']
		module_name = 'm' + '_'.join(module_name.split('_')[1:])
		code_path = os.path.join(module_dir, f"{repo_name}.{module_name}.py")

		code = data['code']
		code = code.replace('Model', module_name)
		with open(code_path, 'w') as f:
			f.write(code)
		log.info(f'Wrote {code_path}')


def get_args(raw_args=None):
	import torch
	import torch._dynamo

	parser = argparse.ArgumentParser()
	group = parser.add_mutually_exclusive_group()
	group.add_argument(
		'--load',
		action='store_true',
		help='load data from GPUMODE/KernelBook dataset ',
		default=True,
	)
	group.add_argument(
		'--reload',
		action='store_true',
		help='reload data from GPUMODE/KernelBook dataset ',
		default=False,
	)
	parser.add_argument(
		'--dataset',
		type=str,
		choices=['kernelbook', 'kernelbench'],
		default='kernelbook',
		help='Dataset to load: kernelbook (GPUMODE/KernelBook) or kernelbench (ScalingIntelligence/KernelBench)',
	)
	parser.add_argument(
		'--kernelbench-level',
		type=str,
		choices=['level_1', 'level_2', 'level_3', 'level_4'],
		default='level_1',
		help='KernelBench difficulty level (only used when --dataset=kernelbench)',
	)
	parser.add_argument(
		'--change_inputs_dims',
		type=int,
		help='change input dimensions for the model (0 or None to skip changing dimensions)',
		default=None,
	)
	parser.add_argument(
		'--repos_file',
		type=str,
		help='file containing list of github repos to download',
		default=None,
	)
	parser.add_argument(
		'--shard_num',
		type=int,
		help='shard number of the current process. This is ignored if repos file is not given or evaluate-all (with a synthetic directory) is not called',
		default=0,
	)
	parser.add_argument(
		'--shard_total',
		type=int,
		help='total number of shards. This is ignored if repos file is not given or evaluate-all (with a synthetic directory) is not called',
		default=1,
	)

	group.add_argument(
		'--evaluate-one', '-e', help='Check torch.jit.script on a given test_*.py file'
	)
	group.add_argument('--evaluate-all', action='store_true', help='Check torch.jit.script parity')

	# TODO: Sahan, put everything (donwload, build, generate, cache) in all one
	parser.add_argument(
		'--run-dir',
		default='./runs/run1',
		help='dir where we have all artifacts for the run (intermediate outputs (ie. download, build, generate, cache) + final dataset)',
	)

	# Number of Parallel Jobs
	parser.add_argument('--jobs', '-j', type=int, default=4)  # Reduced from potentially high values
	parser.add_argument(
		'--offset',
		type=int,
		default=0,
		help='Pick files starting from this offset. Together with --limit, we can run through all files in multiple separate runs',
	)
	parser.add_argument('--limit', '-l', type=int, help='only run the first N files')
	parser.add_argument('--filter', '-f', '-k', help='only run module containing given name')
	parser.add_argument(
		'--no-fork', action='store_true', help="don't run *-one test in a subprocess"
	)
	parser.add_argument('--memory-limit-gb', type=int, default=32)

	parser.add_argument(
		'--onnxdir',
		type=str,
		help='dir where to export modules to onnx during evaluate',
	)
	parser.add_argument(
		'--fullgraph',
		default=False,
		action='store_true',
		help='use fullgraph(no python fall back) when compiling with dynamo',
	)
	parser.add_argument(
		'--compile_mode',
		default='dynamo',
		type=str,
		help='choose a mode of compilation: dynamo, export, aot_inductor or torchscript',
	)
	parser.add_argument(
		'--backend',
		default='inductor',
		type=str,
		help='dynamo backends: {}'.format(torch._dynamo.list_backends()),
	)
	parser.add_argument(
		'--device', default='cuda', type=str, help='evaluate modules using cuda, xpu or cpu'
	)
	parser.add_argument('--metric-path', type=str, help='path of the compilation metric')
	parser.add_argument(
		'--fresh-cache-dir',
		action='store_true',
		help='use a fresh cache dir for each individual inductor test run and remove it after done',
	)
	parser.add_argument(
		'--synthetic-data-dir',
		type=str,
		help='path to the synthetic data directory. This is only used for --evaluate-all',
	)

	args = parser.parse_args(raw_args)
	return args


def main(raw_args=None):
	assert sys.version_info >= (3, 8), 'Python 3.8+ required, got: {}'.format(sys.version)

	import torch
	import torch._dynamo

	args = get_args(raw_args)
	if args.reload:
		args.load = True
	print(args)

	load_dir = args.run_dir + '/load'

	# create directories if they don't exist BEFORE setting up logging
	os.makedirs(args.run_dir, exist_ok=True)
	if args.load:
		os.makedirs(load_dir, exist_ok=True)

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(os.path.join(args.run_dir, 'operator_fusion.log'), mode='w'),  # 'w' overwrites the file
			# logging.StreamHandler()  # Keep console output too
		],
		force=True,  # override any prior logging config from imported modules
	)

	os.environ['RLIMIT_AS_GB'] = str(args.memory_limit_gb)

	# GPU memory management for CUDA
	if args.device == 'cuda':
		import torch
		if torch.cuda.is_available():
			# Enable memory-efficient allocation
			os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
			# Limit GPU memory per process
			torch.cuda.set_per_process_memory_fraction(0.1)  # Use only 10% of GPU memory per process

	if args.load:
		if args.dataset == 'kernelbook':
			log.info('Loading GPUMODE/KernelBook dataset')
			df = pd.read_parquet("hf://datasets/GPUMODE/KernelBook/dataset_permissive.parquet")
			ds = Dataset.from_pandas(df)
			load_code(ds, args.run_dir, args)

		elif args.dataset == 'kernelbench':
			log.info(f'Loading ScalingIntelligence/KernelBench dataset (level: {args.kernelbench_level})')
			# Load with datasets library to avoid PyArrow parquet issues, then convert to pandas
			ds_hf = load_dataset("ScalingIntelligence/KernelBench", split=args.kernelbench_level)
			df = ds_hf.to_pandas()
			ds = Dataset.from_pandas(df)
			load_code_kernel_bench(ds, args.run_dir, args)
		else:
			raise ValueError(f"Unknown dataset: {args.dataset}")

		log.info(f'Loaded {len(ds)} examples from {args.dataset}')
		# ds = ds.select(range(10))

	write_helpers(args.run_dir)

	# args.evaluate_all is the default:
	return evaluate_all(
		args,
		run_dir=args.run_dir,
		offset=args.offset,
		limit=args.limit,
		jobs=args.jobs,
	)


if __name__ == '__main__':
	import sys

	assert sys.version_info >= (3, 8), 'Python 3.8+ required, got: {}'.format(sys.version)

	start = mp.get_start_method(allow_none=True)
	if start != "spawn":
		try:
			mp.set_start_method("spawn", force=True)
		except RuntimeError:
			# Already set by the environment/runner; that's fine.
			pass

	main()
