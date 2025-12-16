#!/usr/bin/env python
"""Test notebook code cells by extracting and executing them."""

import json
import sys
import traceback
from pathlib import Path

def extract_code_cells(notebook_path: Path) -> list[str]:
    """Extract code cells from a notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            # Skip pip install cells
            if '!pip install' in code or code.strip().startswith('#'):
                continue
            code_cells.append(code)
    return code_cells

def test_notebook(notebook_path: Path) -> tuple[bool, str]:
    """Test a notebook by executing its code cells."""
    try:
        cells = extract_code_cells(notebook_path)
        
        # Create a shared namespace
        namespace = {'__name__': '__main__'}
        
        for i, code in enumerate(cells):
            try:
                exec(code, namespace)
            except Exception as e:
                return False, f"Cell {i+1}: {type(e).__name__}: {e}"
        
        return True, "OK"
    except Exception as e:
        return False, f"Parse error: {e}"

def main():
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    notebooks = sorted(notebooks_dir.glob('[0-9][0-9]_*.ipynb'))
    
    print(f"Testing {len(notebooks)} notebooks...\n")
    
    passed = 0
    failed = 0
    errors = []
    
    for nb_path in notebooks:
        success, msg = test_notebook(nb_path)
        if success:
            passed += 1
            print(f"PASS: {nb_path.name}")
        else:
            failed += 1
            errors.append((nb_path.name, msg))
            print(f"FAIL: {nb_path.name} - {msg}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if errors:
        print(f"\nErrors to fix:")
        for name, err in errors:
            print(f"  {name}: {err}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
