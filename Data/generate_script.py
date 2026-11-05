from datasets import load_dataset, Dataset
import argparse
import logging
import os
import sys
import multiprocessing as mp
from functools import partial
import pandas as pd


import torch
import torch._dynamo
from tqdm.auto import tqdm as tqmd

def load_code(ds, run_dir):
	for data in tqmd(ds):
		module_name = data['module_name']
		module_dir = os.path.join(run_dir, 'pytorch_modules')
		triton_dir = os.path.join(run_dir, 'triton_modules')
		os.makedirs(triton_dir, exist_ok=True)
		os.makedirs(module_dir, exist_ok=True)
		py_path = os.path.join(module_dir, data['module_name'] + '.py')
		triton_path = os.path.join(triton_dir, data['module_name'] + '_triton.py')

		if os.path.exists(py_path):
			continue

		if not os.path.exists(py_path):
			code = data['python_code']
			with open(py_path, 'w') as f:
				f.write(code)

			triton_code = data['triton_code']
			with open(triton_path, 'w') as f:
				f.write(triton_code)

def get_args(raw_args=None):

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--run-dir',
		default='./runs/run1',
		help='dir where we have all artifacts for the run (intermediate outputs (ie. download, build, generate, cache) + final dataset)',
	)
	parser.add_argument(
		'--data-dir',
		default='./data/',
	)
	args = parser.parse_args(raw_args)
	return args

def main(raw_args=None):
	assert sys.version_info >= (3, 8), 'Python 3.8+ required, got: {}'.format(sys.version)
	args = get_args(raw_args)
	script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_dir)
	os.makedirs(args.data_dir, exist_ok=True)
	# script_dir = os.path.dirname(os.path.abspath(__file__))

	dataset_path = os.path.join(args.run_dir, 'datasets', 'dataset.parquet')
	if not os.path.exists(dataset_path):
		raise FileNotFoundError(f'Dataset not found at {dataset_path}')
	df = pd.read_parquet(dataset_path)
	ds = Dataset.from_pandas(df)	# ds['train'] = ds['train'].select(range(10))
	load_code(ds, script_dir)


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
