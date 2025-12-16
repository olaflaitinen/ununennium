#!/usr/bin/env python
"""Run a subset of notebooks for CI smoke testing.

Executes 5 representative notebooks to verify basic functionality.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Representative notebooks for smoke testing
SMOKE_NOTEBOOKS = [
    "00_quickstart.ipynb",
    "10_spectral_indices.ipynb",
    "20_training_basics.ipynb",
    "35_pinn_fundamentals.ipynb",
    "46_benchmarking.ipynb",
]


def run_notebook(notebook_path: Path, timeout: int = 300) -> bool:
    """Execute a notebook and return success status.

    Args:
        notebook_path: Path to the notebook file.
        timeout: Maximum execution time in seconds.

    Returns:
        True if notebook executed successfully, False otherwise.
    """
    print(f"Running: {notebook_path.name}")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout",
                str(timeout),
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )

        if result.returncode == 0:
            print(f"  PASSED: {notebook_path.name}")
            return True
        else:
            print(f"  FAILED: {notebook_path.name}")
            print(f"  Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {notebook_path.name}")
        return False
    except Exception as e:
        print(f"  ERROR: {notebook_path.name} - {e}")
        return False


def main() -> int:
    """Run smoke test notebooks.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Run notebook smoke tests")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per notebook in seconds (default: 300)",
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path(__file__).parent.parent / "notebooks",
        help="Path to notebooks directory",
    )
    args = parser.parse_args()

    notebooks_dir = args.notebooks_dir
    if not notebooks_dir.exists():
        print(f"Notebooks directory not found: {notebooks_dir}")
        return 1

    print(f"Running smoke tests on {len(SMOKE_NOTEBOOKS)} notebooks")
    print("=" * 60)

    passed = 0
    failed = 0

    for notebook_name in SMOKE_NOTEBOOKS:
        notebook_path = notebooks_dir / notebook_name
        if not notebook_path.exists():
            print(f"  MISSING: {notebook_name}")
            failed += 1
            continue

        if run_notebook(notebook_path, args.timeout):
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
