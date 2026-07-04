"""Generic verifier: execute every code cell of an .ipynb in one namespace.

Usage:  PYTHONPATH=. python3 notebooks/_build/_verify_nb.py 02_allyl_uj_identity.ipynb
"""
import os, sys, time
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')

if len(sys.argv) < 2:
    sys.exit("usage: _verify_nb.py <notebook>.ipynb")

nb_name = sys.argv[1]
nb_path = os.path.join(os.path.dirname(__file__), '..', nb_name)
nb = nbf.read(nb_path, as_version=4)

ns = {'__name__': '__verify__'}
n_code = 0
t0 = time.time()
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    n_code += 1
    print(f"--- cell {i} (code #{n_code}) ---")
    try:
        exec(compile(cell.source, f'<cell {i}>', 'exec'), ns)
    except Exception:
        print(f"FAILED in cell {i}:\n{cell.source}\n")
        raise
print(f"\nAll {n_code} code cells executed cleanly in {time.time()-t0:.1f}s.")
