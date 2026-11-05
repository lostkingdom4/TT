import json
import re
import glob

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os  # To remove the temporary file after processing
import subprocess  # To run external commands (i.e. Ruff)
import tempfile  # To create a temporary file to store the code
import logging
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU architecture mapping
# ---------------------------------------------------------------------------

GPU_ARCH_MAP = {
    "v100": "cuda:70",
    "a100": "cuda:80",
    "rtx3090": "cuda:86",
    "a40": "cuda:a40",
    "l40s": "cuda:l40s",
    "h100": "cuda:90",
}


# ---------------------------------------------------------------------------
# IR discovery helpers
# ---------------------------------------------------------------------------

def get_device_name(ir_output_dir: Path) -> str:
    """Extract device name from an IR output path.

    Example: extract/ir_output/v100 -> 'v100'
             extract/ir_output/A40  -> 'A40'
    """
    return Path(ir_output_dir).resolve().name


def extract_input_index(filename: str) -> int:
    """Extract input index from a filename like '*_input5_shapes.json'."""
    m = re.search(r"_input(\d+)_shapes\.json$", filename)
    return int(m.group(1)) if m else 0


def parse_shapes_file(path: Path) -> Optional[Dict]:
    """Load a shapes JSON and extract key metadata.

    Returns a dict with kernel_name, input_index, total_args, has_grid,
    and kwargs_keys, or None on parse failure.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "kernel_name": data.get("kernel", "unknown"),
            "input_index": data.get("input_index", extract_input_index(Path(path).name)),
            "total_args": data.get("total_args", 0),
            "has_grid": "grid" in data,
            "kwargs_keys": list(data.get("kwargs", {}).keys()),
        }
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to parse %s: %s", path, e)
        return None


def run_ruff_on_code(code: str) -> str:
	"""
	Run Ruff on a given Python code string and return its output.

	This function writes the provided Python code into a temporary file,
	calls Ruff on that file using subprocess, captures the output, and then
	removes the temporary file.

	Parameters:
		code (str): The Python source code to lint.

	Returns:
		str: The stdout output from Ruff containing linting messages.
	"""
	# Create a temporary file with a .py suffix so Ruff treats it as Python code.
	with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
		tmp_file.write(code)  # Write the code string to the file.
		tmp_filename = tmp_file.name  # Save the temporary file's name for later use.

	try:
		# Run Ruff on the temporary file.
		# - "ruff" should be installed and available in your system's PATH.
		# - capture_output=True and text=True ensure we capture stdout as a string.
		result = subprocess.run(
			[
				'ruff',
				'check',
				tmp_filename,
				'--fix',
				'--unsafe-fixes',
				'--fix-only',
			],
			capture_output=True,
			text=True,
		)
		output = open(tmp_filename).read()
	finally:
		# Remove the temporary file whether or not Ruff runs successfully.
		os.remove(tmp_filename)

	return output


def lint_code_directory(directory: str) -> None:
	"""
	Apply ruff linter to a directory of code files.

	Args:
		directory: Path to the directory to lint
	"""
	print(f'Linting code in {directory}')
	subprocess.run([
		'ruff',
		'check',
		directory,
		'--fix',
		'--unsafe-fixes',
		'--fix-only',
	])


def replace_number_in_functions(directory: str, replacement_value: str) -> None:
	"""
	Replace all occurrences of the number 4 in get_inputs and get_init_inputs functions
	with a specified replacement value in all Python files in the given directory.

	Args:
		directory: Path to the directory containing Python files to process
		replacement_value: The value to replace all occurrences of number 4 with
	"""

	# Find all Python files in the directory
	python_files = glob.glob(os.path.join(directory, "*.py"))

	modified_files = []
	unmodified_files = []

	for file_path in python_files:
		try:
			with open(file_path, 'r', encoding='utf-8') as f:
				content = f.read()

			log.debug(f"Processing file: {file_path}")

			# Fixed pattern to match Python function bodies properly
			# This regex finds function definitions and captures everything until the next function or end of file
			pattern = r'(def\s+(get_inputs|get_init_inputs)\s*\([^)]*\):[\s\S]*?)(?=\ndef\s+\w+|\Z)'

			# Check if pattern finds any matches
			matches = re.findall(pattern, content, flags=re.MULTILINE)
			log.debug(f"Found {len(matches)} function matches in {file_path}")

			if matches:
				for i, (full_match, func_name) in enumerate(matches):
					log.debug(f"Match {i+1}: Function '{func_name}' - Length: {len(full_match)} chars")
					# Show first 100 chars of match for debugging
					preview = full_match.replace('\n', '\\n')[:100]
					log.debug(f"Preview: {preview}...")

			def replace_in_function(match):
				function_body = match.group(1)
				func_name = match.group(2)
				log.debug(f"Replacing numbers in function: {func_name}")

				# Count occurrences before replacement
				count_before = len(re.findall(r'\b4\b', function_body))
				log.debug(f"Found {count_before} occurrences of '4' to replace")

				# Replace standalone number 4 with replacement value
				# Use word boundaries to avoid replacing 4 in larger numbers like 14, 40, etc.
				modified_body = re.sub(r'\b4\b', replacement_value, function_body)

				# Count after replacement
				count_after = len(re.findall(r'\b4\b', modified_body))
				log.debug(f"After replacement: {count_after} occurrences of '4' remain")

				return modified_body

			# Apply the replacement
			modified_content = re.sub(pattern, replace_in_function, content, flags=re.MULTILINE)

			if modified_content != content:
				log.debug(f"Content modified for {file_path}")
			else:
				log.debug(f"No changes made to {file_path}")
			# Write back to file if content changed
			if modified_content != content:
				with open(file_path, 'w', encoding='utf-8') as f:
					f.write(modified_content)
				modified_files.append(file_path)
			else:
				unmodified_files.append(file_path)

		except Exception as e:
			log.error(f"Error processing {file_path}: {e}")
			unmodified_files.append(file_path)

	# Log summary of modifications
	log.info(f"Modified {len(modified_files)} files, {len(unmodified_files)} files unchanged")
	log.debug(f"Modified files: {modified_files}")
	log.debug(f"Unmodified files: {unmodified_files}")