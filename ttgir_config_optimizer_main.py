#!/usr/bin/env python3
"""
TTGIR Configuration Optimizer - Main Script

Analyzes Triton GPU IR (TTGIR) and metadata to propose optimal kernel configurations
for a given GPU architecture, including memory access patterns and cache behavior.
"""

import json
import argparse
import logging
from pathlib import Path

from optimizer_functions import (
	analyze_kernel_from_cache,
	optimize_kernel_from_cache,
	extract_best_config,
	# benchmark_baseline_config,
	test_recommended_config
)

from configuration_optimizer import ConfigurationOptimizer

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
	parser = argparse.ArgumentParser(
		description="Analyze TTGIR and propose optimal kernel configurations"
	)
	parser.add_argument(
		"--cache-dir",
		type=Path,
		default=Path("./captured_ir/triton_cache"),
		help="Path to Triton cache directory"
	)
	parser.add_argument(
		"--triton-modules-dir",
		type=Path,
		default=Path("./Data/data_64/triton_modules"),
		help="Path to Triton modules directory with original source code"
	)
	parser.add_argument(
		"--target-gpu",
		type=str,
		default="cuda:70",
		choices=["cuda:70", "cuda:80", "cuda:86", "cuda:a40", "cuda:l40s", "cuda:90"],
		help="Target GPU architecture (V100/A100/RTX3090/A40/L40S/H100)"
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Output JSON file for results (default: print to stdout)"
	)
	parser.add_argument(
		"--analyze-all",
		action="store_true",
		help="Analyze all kernels in cache directory"
	)
	parser.add_argument(
		"--test-config",
		action="store_true",
		help="Test the best recommended configuration with actual kernel execution"
	)
	parser.add_argument(
		"--input-shapes-json",
		type=Path,
		help="Path to *_shapes.json file from capture_all_kernel_ir.py (required for --test-config)"
	)
	parser.add_argument(
		"--test-device",
		type=str,
		default="cuda",
		choices=["cuda", "xpu", "cpu"],
		help="Device to use for testing (default: cuda)"
	)
	parser.add_argument(
		"--num-iterations",
		type=int,
		default=100,
		help="Number of benchmark iterations when testing config (default: 100)"
	)
	parser.add_argument(
		"--benchmark-baseline",
		action="store_true",
		help="Benchmark the baseline (default) configuration before testing recommended config"
	)
	parser.add_argument(
		"--input-shapes-dir",
		type=Path,
		default=Path("./captured_ir/input_shapes"),
		help="Directory containing *_shapes.json files (default: ./captured_ir/input_shapes)"
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		help="Directory to save individual kernel results (only with --analyze-all)"
	)
	parser.add_argument(
		"--limit",
		type=int,
		help="Limit number of kernels to process when using --analyze-all (default: process all)"
	)
	parser.add_argument(
		"--model",
		type=str,
		default="deepseek-chat",
		help="DeepSeek model to use (default: deepseek-chat, alternatives: deepseek-reasoner)"
	)

	# Multi-agent system options
	multiagent_group = parser.add_argument_group(
		'multi-agent optimization options',
		'Options for using the LangGraph multi-agent optimization system'
	)
	multiagent_group.add_argument(
		"--use-multiagent",
		action="store_true",
		help="Use LangGraph multi-agent system instead of single-agent ConfigurationOptimizer"
	)
	multiagent_group.add_argument(
		"--llm-provider",
		type=str,
		default="deepseek",
		choices=["deepseek", "openai", "ollama", "vllm"],
		help="LLM provider for multi-agent system (default: deepseek)"
	)
	multiagent_group.add_argument(
		"--llm-model",
		type=str,
		help="Specific LLM model name (optional, uses provider default)"
	)
	multiagent_group.add_argument(
		"--temperature",
		type=float,
		default=0.2,
		help="LLM temperature for multi-agent system (default: 0.2)"
	)
	multiagent_group.add_argument(
		"--max-iterations",
		type=int,
		default=1,
		help="Max refinement iterations for multi-agent system (default: 1)"
	)
	multiagent_group.add_argument(
		"--optimization-target",
		type=str,
		default="throughput",
		choices=["throughput", "latency", "energy"],
		help="Optimization target for multi-agent system (default: throughput)"
	)
	multiagent_group.add_argument(
		"--debug-mode",
		action="store_true",
		help="Enable debug mode: saves all prompts and responses to files"
	)
	multiagent_group.add_argument(
		"--debug-dir",
		type=Path,
		default=Path("./debug_multiagent"),
		help="Directory to save debug artifacts (default: ./debug_multiagent/)"
	)

	# Cached response group - all three must be provided together
	cached_group = parser.add_argument_group(
		'cached response options',
		'All three arguments must be provided together when using cached responses'
	)
	cached_group.add_argument(
		"--kernel-name",
		type=str,
		help="Kernel name to analyze (e.g., 'triton_poi_fused_add_convolution_2')"
	)
	cached_group.add_argument(
		"--kernel-file",
		type=Path,
		help="Path to the kernel Python file (optional, will search if not provided)"
	)
	cached_group.add_argument(
		"--torch-model-name",
		type=str,
		help="Name of the Torch model (required with cached response)"
	)
	cached_group.add_argument(
		"--use-cached-response",
		action="store_true",
		help="Use cached JSON response instead of calling API"
	)
	cached_group.add_argument(
		"--cached-response-file",
		type=Path,
		help="Path to cached JSON response file (required with --use-cached-response)"
	)
	cached_group.add_argument(
		"--prompt-output-dir",
		type=Path,
		default=Path("./generated_prompts"),
		help="Directory to save generated prompts organized by torch_model/kernel_name"
	)

	args = parser.parse_args()

	if args.analyze_all:
		# Find all TTGIR files
		ttgir_files = sorted(list(args.cache_dir.rglob("*.ttgir")))
		print(f"Found {len(ttgir_files)} TTGIR files to analyze")

		# Apply limit if specified
		if args.limit and args.limit > 0:
			ttgir_files = ttgir_files[:args.limit]
			print(f"Limiting to first {args.limit} kernels\n")
		else:
			print()

		# Create output directory if specified
		if args.output_dir:
			args.output_dir.mkdir(parents=True, exist_ok=True)
			print(f"Results will be saved to: {args.output_dir}\n")
			prompt_output_dir = args.output_dir / "generated_prompts"
			prompt_output_dir.mkdir(parents=True, exist_ok=True)
		else:
			raise ValueError("When using --analyze-all, --output-dir must be specified to save results")

		results = []
		success_count = 0
		failure_count = 0

		for idx, ttgir_path in enumerate(ttgir_files, 1):
			kernel_name = ttgir_path.stem

			# Extract source file info early for tracking
			source_file_info = None
			ttgir_content = ttgir_path.read_text()
			source_file, line_range = ConfigurationOptimizer._extract_source_location(ttgir_content)
			if source_file:
				source_file_info = {
					"source_file": str(source_file),
					"line_range": line_range
				}

			if source_file_info:
				# Hash source file path to create unique suffix
				source_model = source_file_info["source_file"].split('/')[-1].split('.')[0]
				unique_kernel_id = f"{kernel_name}_{source_model}"
			else:
				raise ValueError(f"Could not extract source file info from TTGIR: {ttgir_path}")

			print(f"\n{'='*80}")
			print(f"[{idx}/{len(ttgir_files)}] Processing: {kernel_name}")
			if source_file_info:
				print(f"Source: {source_model}")
				if source_file_info.get('line_range'):
					print(f"Lines: {source_file_info['line_range'][0]}-{source_file_info['line_range'][1]}")
				print(f"Unique ID: {unique_kernel_id}")
			print('='*80)

			try:
				# Try to find input shapes JSON file
				input_shapes_json = None

				# Search for matching shapes file in input_shapes_dir
				if args.input_shapes_dir and args.input_shapes_dir.exists():
					# Try multiple naming patterns
					patterns = [
						f"*{kernel_name}_shapes.json",
						f"{kernel_name}_shapes.json",
						f"*_{kernel_name}_shapes.json"
					]

					for pattern in patterns:
						matches = list(args.input_shapes_dir.glob(pattern))
						if matches:
							input_shapes_json = matches[0]
							print(f"\nFound input shapes: {input_shapes_json}")
							break

				if not input_shapes_json:
					raise FileNotFoundError(f"No input shapes JSON found for kernel: {kernel_name}")

				# Choose optimization method based on flag
				if args.use_multiagent:
					# Use multi-agent system
					result = optimize_kernel_from_cache(
						cache_dir=args.cache_dir,
						kernel_name=kernel_name,
						torch_model=source_model,
						input_shapes_json=input_shapes_json,
						target_gpu=args.target_gpu,
						triton_modules_dir=args.triton_modules_dir,
						llm_provider=args.llm_provider,
						llm_model=args.llm_model,
						temperature=args.temperature,
						max_iterations=args.max_iterations,
						optimization_target=args.optimization_target,
						debug_mode=args.debug_mode,
						debug_dir=args.debug_dir,
					)
				else:
					# Use traditional single-agent ConfigurationOptimizer
					result = analyze_kernel_from_cache(
						args.cache_dir, kernel_name, source_model, input_shapes_json,
						args.target_gpu, args.triton_modules_dir,
						args.use_cached_response, args.cached_response_file,
						args.model, prompt_output_dir,
					)
				# Add source file info to result
				if result and source_file_info:
					result["source_file_info"] = source_file_info
					result["unique_kernel_id"] = unique_kernel_id

				if not result:
					print(f"Failed to analyze {kernel_name}")
					failure_count += 1
					continue

				# Test configuration if requested
				if args.test_config:
					# Find kernel source file
					kernel_file = None

					# Strategy 1: Extract from TTGIR
					if source_file:
						source_path = Path(source_file)
						if source_path.exists():
							kernel_file = source_path
							print(f"Found kernel file: {kernel_file}")

					if not kernel_file:
						print(f"\nWarning: Could not find kernel source file for {kernel_name}, skipping benchmarks")
						results.append(result)
						continue

					# Extract best config
					best_config = extract_best_config(result)

					if not best_config:
						print(f"\nWarning: Could not extract best config for {kernel_name}, skipping benchmarks")
						results.append(result)
						continue

					# Benchmark baseline if requested
					# if args.benchmark_baseline:
					# 	try:
					# 		print(f"\nBenchmarking baseline configuration...")
					# 		baseline_result = benchmark_baseline_config(
					# 			kernel_name=kernel_name,
					# 			kernel_path=kernel_file,
					# 			input_shapes_json=input_shapes_json,
					# 			device=args.test_device,
					# 			num_iterations=args.num_iterations
					# 		)
					# 		result["baseline_benchmark"] = baseline_result
					# 	except Exception as e:
					# 		print(f"Error benchmarking baseline: {e}")
					# 		result["baseline_benchmark"] = {"success": False, "error": str(e)}

					# Test recommended configuration
					try:
						print(f"\nBenchmarking recommended configuration...")
						test_result = test_recommended_config(
							kernel_name=kernel_name,
							kernel_path=kernel_file,
							input_shapes_json=input_shapes_json,
							config=best_config,
							device=args.test_device,
							num_iterations=args.num_iterations
						)
						result["optimized_benchmark"] = test_result

						# Calculate speedup if baseline was benchmarked
						if args.benchmark_baseline and result.get("baseline_benchmark", {}).get("success") and test_result.get("success"):
							baseline_time = result["baseline_benchmark"]["execution_time_ms"]["median"]
							optimized_time = test_result["execution_time_ms"]["median"]
							speedup = baseline_time / optimized_time if optimized_time > 0 else 0

							result["performance_comparison"] = {
								"baseline_time_ms": baseline_time,
								"optimized_time_ms": optimized_time,
								"speedup": speedup,
								"improvement_percent": (speedup - 1) * 100
							}

							print(f"\n{'='*80}")
							print("Performance Comparison:")
							print(f"  Baseline time: {baseline_time:.4f} ms")
							print(f"  Optimized time: {optimized_time:.4f} ms")
							print(f"  Speedup: {speedup:.2f}x")
							print(f"  Improvement: {(speedup - 1) * 100:.2f}%")
							print(f"{'='*80}\n")
					except Exception as e:
						print(f"Error benchmarking optimized config: {e}")
						result["optimized_benchmark"] = {"success": False, "error": str(e)}

				results.append(result)
				success_count += 1

				# Save individual result if output_dir specified
				if args.output_dir:
					output_file = args.output_dir / f"{unique_kernel_id}_result.json"
					with open(output_file, 'w') as f:
						json.dump(result, f, indent=2)
					print(f"Saved result to: {output_file}")

			except Exception as e:
				print(f"\nError processing {kernel_name}: {e}")
				import traceback
				traceback.print_exc()
				failure_count += 1

				# Save error result
				error_result = {
					"kernel_name": kernel_name,
					"unique_kernel_id": unique_kernel_id,
					"success": False,
					"error": str(e)
				}
				if source_file_info:
					error_result["source_file_info"] = source_file_info
				results.append(error_result)

				if args.output_dir:
					output_file = args.output_dir / f"{unique_kernel_id}_error.json"
					with open(output_file, 'w') as f:
						json.dump(error_result, f, indent=2)

		# Summary
		print(f"\n{'='*80}")
		print(f"Batch Processing Complete")
		print(f"{'='*80}")
		print(f"Total kernels: {len(ttgir_files)}")
		print(f"Successful: {success_count}")
		print(f"Failed: {failure_count}")
		print(f"{'='*80}\n")

		# Save combined results if output file specified
		if args.output:
			with open(args.output, 'w') as f:
				json.dump(results, f, indent=2)
			print(f"Combined results saved to: {args.output}\n")

	elif args.kernel_name and args.torch_model_name:
		# Validate cached response arguments - all three must be provided together
		if args.use_cached_response and not args.cached_response_file:
			print("Error: --cached-response-file is required when using --use-cached-response")
			print("\nWhen using cached responses, you must provide:")
			print("  --torch-model-name <name>")
			print("  --use-cached-response")
			print("  --cached-response-file <path>")
			return 1

		if not args.cache_dir:
			print("Error: --cache-dir is required when analyzing a specific kernel")
			print(f"\nPlease provide the directory containing TTGIR files for model '{args.torch_model_name}'")
			print("Example: --cache-dir ./triton_cache/{args.torch_model_name}/")
			return 1

		if not args.cache_dir.exists():
			print(f"Error: Cache directory does not exist: {args.cache_dir}")
			print(f"\nExpected directory for model '{args.torch_model_name}': {args.cache_dir}")
			print("Make sure you've captured TTGIR files for this model first.")
			return 1


		# Try to find input shapes JSON file
		input_shapes_json = args.input_shapes_json

		if not input_shapes_json:
			print(f"\nAttempting to auto-find input shapes JSON...")
			if args.input_shapes_dir and args.input_shapes_dir.exists():
				patterns = [
					f"*{args.kernel_name}_shapes.json",
					f"{args.kernel_name}_shapes.json",
					f"*_{args.kernel_name}_shapes.json"
				]

				for pattern in patterns:
					matches = list(args.input_shapes_dir.glob(pattern))
					if matches:
						input_shapes_json = matches[0]
						print(f"Found input shapes: {input_shapes_json}")
						break

		if not input_shapes_json:
			raise FileNotFoundError(f"No input shapes JSON found for kernel: {args.kernel_name}")

		# Choose optimization method based on flag
		if args.use_multiagent:
			# Use multi-agent system
			result = optimize_kernel_from_cache(
				cache_dir=args.cache_dir,
				kernel_name=args.kernel_name,
				torch_model=args.torch_model_name,
				input_shapes_json=input_shapes_json,
				target_gpu=args.target_gpu,
				triton_modules_dir=args.triton_modules_dir,
				llm_provider=args.llm_provider,
				llm_model=args.llm_model,
				temperature=args.temperature,
				max_iterations=args.max_iterations,
				optimization_target=args.optimization_target,
				debug_mode=args.debug_mode,
				debug_dir=args.debug_dir,
			)
		else:
			# Use traditional single-agent ConfigurationOptimizer
			result = analyze_kernel_from_cache(
				args.cache_dir, args.kernel_name, args.torch_model_name, input_shapes_json,
				args.target_gpu, args.triton_modules_dir,
				args.use_cached_response, args.cached_response_file,
				args.model, args.prompt_output_dir
			)
		# Add source file info to result
		if result:
			if args.output:
				with open(args.output, 'w') as f:
					json.dump(result, f, indent=2)
				print(f"Results saved to: {args.output}")
			else:
				print("\n" + json.dumps(result, indent=2))

			# Test the recommended configuration if requested
			if args.test_config:
				print(f"\nUsing input shapes JSON: {input_shapes_json}")

				# Extract best config
				best_config = extract_best_config(result)

				if best_config:
					# Find the kernel Python file - try multiple search strategies
					kernel_file = None

					# If user provided kernel file path, use it
					if args.kernel_file:
						if args.kernel_file.exists():
							kernel_file = args.kernel_file
							print(f"Using provided kernel file: {kernel_file}")
						else:
							print(f"Error: Provided kernel file does not exist: {args.kernel_file}")
							return 1
					else:
						# Strategy 1: Extract source location from TTGIR file
						print(f"\nFinding kernel source file from TTGIR...")
						ttgir_files = list(args.cache_dir.rglob(f"*{args.kernel_name}.ttgir"))
						if ttgir_files:
							for ttgir_path in ttgir_files:
								with open(ttgir_path, 'r') as f:
									ttgir_content = f.read()
								# Use ConfigurationOptimizer's method to extract source location
								source_file, line_range = ConfigurationOptimizer._extract_source_location(ttgir_content)
								source = source_file.split('/')[-1].split('.')[0]

								if source_file and source == args.torch_model_name:
									source_path = Path(source_file)
									if source_path.exists():
										kernel_file = source_path
										print(f"Found kernel from TTGIR loc info: {kernel_file}")
										if line_range:
											print(f"  Line range: {line_range[0]}-{line_range[1]}")
									else:
										print(f"Source file from TTGIR doesn't exist: {source_file}")
								else:
									print("Could not extract source location from TTGIR")


					if not kernel_file:
						print(f"\nError: Could not find kernel file for '{args.kernel_name}'")
						print(f"Searched in:")
						print(f"  - TTGIR loc attributes")
						print(f"  - {args.triton_modules_dir}")
						print(f"  - {args.triton_modules_dir.parent}")
						print("\nPlease provide the kernel file path manually using --kernel-file")
						print("or ensure the kernel file exists in the expected location.")
						return 1

					# Benchmark baseline configuration if requested
					baseline_result = {}
					# if args.benchmark_baseline:
					# 	baseline_result = benchmark_baseline_config(
					# 		kernel_name=args.kernel_name,
					# 		kernel_path=kernel_file,
					# 		input_shapes_json=input_shapes_json,
					# 		device=args.test_device,
					# 		num_iterations=args.num_iterations
					# 	)
					# 	result["baseline_benchmark"] = baseline_result

					# Test the recommended configuration
					test_result = test_recommended_config(
						kernel_name=args.kernel_name,
						kernel_path=kernel_file,
						input_shapes_json=input_shapes_json,
						config=best_config,
						device=args.test_device,
						num_iterations=args.num_iterations
					)

					# Add test results to the overall result
					result["optimized_benchmark"] = test_result

					# Calculate speedup if baseline was benchmarked
					if args.benchmark_baseline and baseline_result.get("success") and test_result.get("success"):
						baseline_time = baseline_result["execution_time_ms"]["median"]
						optimized_time = test_result["execution_time_ms"]["median"]
						speedup = baseline_time / optimized_time if optimized_time > 0 else 0

						result["performance_comparison"] = {
							"baseline_time_ms": baseline_time,
							"optimized_time_ms": optimized_time,
							"speedup": speedup,
							"improvement_percent": (speedup - 1) * 100
						}

						print(f"\n{'='*80}")
						print("Performance Comparison:")
						print(f"  Baseline time: {baseline_time:.4f} ms")
						print(f"  Optimized time: {optimized_time:.4f} ms")
						print(f"  Speedup: {speedup:.2f}x")
						print(f"  Improvement: {(speedup - 1) * 100:.2f}%")
						print(f"{'='*80}\n")

					# Save updated results if output file specified
					if args.output:
						with open(args.output, 'w') as f:
							json.dump(result, f, indent=2)
						print(f"\nUpdated results (including benchmark) saved to: {args.output}")
				else:
					print("\nCould not extract best configuration. Skipping test.")

	else:
		parser.print_help()
		print("\nError: Must specify either (--kernel-name and --torch-model-name) or --analyze-all")
		return 1

	return 0


if __name__ == "__main__":
	exit(main())