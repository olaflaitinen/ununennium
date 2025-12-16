#!/usr/bin/env python
"""Run all notebooks for comprehensive local testing.

Executes all 50 notebooks in the notebooks directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_notebook(notebook_path: Path, timeout: int = 600) -> bool:
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
    """Run all notebooks.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Run all notebooks")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per notebook in seconds (default: 600)",
    )
    parser.add_argument(
        "--notebooks-dir",
        type=Path,
        default=Path(__file__).parent.parent / "notebooks",
        help="Path to notebooks directory",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop execution on first failure",
    )
    args = parser.parse_args()

    notebooks_dir = args.notebooks_dir
    if not notebooks_dir.exists():
        print(f"Notebooks directory not found: {notebooks_dir}")
        return 1

    # Find all notebooks sorted by name
    notebooks = sorted(notebooks_dir.glob("[0-9][0-9]_*.ipynb"))

    if not notebooks:
        print("No notebooks found")
        return 1

    print(f"Running all {len(notebooks)} notebooks")
    print("=" * 60)

    passed = 0
    failed = 0
    failed_notebooks = []

    for notebook_path in notebooks:
        if run_notebook(notebook_path, args.timeout):
            passed += 1
        else:
            failed += 1
            failed_notebooks.append(notebook_path.name)
            if args.stop_on_failure:
                print("Stopping on first failure")
                break

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(notebooks)}")

    if failed_notebooks:
        print("\nFailed notebooks:")
        for name in failed_notebooks:
            print(f"  - {name}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
