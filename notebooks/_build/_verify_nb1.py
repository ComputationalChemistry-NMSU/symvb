"""Execute every code cell of 01_h2_from_scratch.ipynb in order, in one
namespace, to verify the notebook runs without errors.

Usage:  PYTHONPATH=. python3 notebooks/_build/_verify_nb1.py
"""
import os
import sys
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')   # no display

nb_path = os.path.join(os.path.dirname(__file__), '..', '01_h2_from_scratch.ipynb')
nb = nbf.read(nb_path, as_version=4)

ns = {'__name__': '__verify__'}
n_code = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    n_code += 1
    src = cell.source
    print(f"--- cell {i} (code #{n_code}) ---")
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
    except Exception as e:
        print(f"FAILED in cell {i}:\n{src}\n")
        raise
print(f"\nAll {n_code} code cells executed cleanly.")
