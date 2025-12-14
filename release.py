"""Automated release script for Ununennium."""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def clean():
    print("Cleaning build artifacts...")
    patterns = ["build", "dist", "*.egg-info", "__pycache__"]
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
    print("Cleaned.")

def run_tests():
    print("Running tests...")
    import subprocess
    ret = subprocess.call(f"{sys.executable} -m pytest tests/", shell=True)
    if ret != 0:
        print(f"ERROR: Tests failed with exit code {ret}.")
        sys.exit(ret)
    else:
        print("Tests passed!")

def build():
    print("Building package...")
    run_command(f"{sys.executable} -m build")

def verify_structure():
    print("Verifying directory structure...")
    required = [
        "src/ununennium",
        "tests",
        "docs",
        "examples",
        "pyproject.toml",
        "README.md",
        "LICENSE"
    ]
    for req in required:
        if not os.path.exists(req):
            raise FileNotFoundError(f"Missing required file/directory: {req}")
    print("Structure verification passed.")

def main():
    print("Starting Ununennium Release Process")
    print("=" * 40)
    
    try:
        clean()
        verify_structure()
        run_tests()
        build()
        
        print("\n" + "=" * 40)
        print("Release build successful!")
        print("Artifacts are in 'dist/' directory.")
        print("To publish to PyPI, run: twine upload dist/*")
        
    except Exception as e:
        print(f"\nRelease failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
