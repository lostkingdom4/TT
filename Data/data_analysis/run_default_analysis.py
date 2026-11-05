#!/usr/bin/env python3
"""
Simple runner for default configuration analysis

Usage:
    python run_default_analysis.py
    python run_default_analysis.py --verbose
    python run_default_analysis.py --data-dir ../data_64 --output my_results.json
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the default configuration analyzer"""

    # Path to the analyzer script
    analyzer_script = Path(__file__).parent / "default_config_analyzer.py"

    if not analyzer_script.exists():
        print(f"❌ Analyzer script not found: {analyzer_script}")
        return 1

    # Forward all arguments to the analyzer
    cmd = [sys.executable, str(analyzer_script)] + sys.argv[1:]

    try:
        # Run the analyzer
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())