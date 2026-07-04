"""
Benchmark the two o2_det implementations on a fixed system battery.

Runs each (system, method) combination N_TRIALS times, reports the median
of wall-time and peak memory, plus the total SymPy-op count of the
resulting matrix as a proxy for output expression complexity.

Usage:
    cd vbt-3-repo
    PYTHONPATH=. python3 -m symvb._o2_benchmark
"""
from __future__ import annotations

import gc
import time
import tracemalloc

import sympy as sp

from symvb import Molecule


# --- Cases ---------------------------------------------------------------

_PPP_2E = {'U': ('1111',), 'J': ('1212',), 'K': ('1122',),
           'M': ('1112', '1121', '1222')}


def _h2(cutoff):
    return dict(
        label='H2',
        Na=1, Nb=1, Norbs=2,
        kwargs=dict(subst_2e=_PPP_2E,
                    interacting_orbs=['ab'],
                    max_2e_centers=cutoff),
    )


def _allyl(cutoff):
    return dict(
        label='allyl-3c4e',
        Na=2, Nb=2, Norbs=3,
        kwargs=dict(subst_2e=_PPP_2E,
                    interacting_orbs=['ab', 'bc'],
                    max_2e_centers=cutoff),
    )


def _benzene_ppp():
    return dict(
        label='benzene-PPP',
        Na=3, Nb=3, Norbs=6,
        kwargs=dict(
            zero_ii=True,
            interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'],
            subst={'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af'),
                   's': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af')},
            subst_2e=_PPP_2E,
            max_2e_centers=2,
        ),
    )


CASES = [
    (_h2(cutoff=1), 5),
    (_h2(cutoff=2), 5),
    (_allyl(cutoff=1), 5),
    (_allyl(cutoff=2), 5),
    (_allyl(cutoff=3), 3),
    (_allyl(cutoff=4), 3),
    (_benzene_ppp(), 1),  # 1 trial; if blocked is fast we can rerun with more
]


# --- Bench runner --------------------------------------------------------

def _build_matrix_o2(method, case):
    m = Molecule(o2_method=method, **case['kwargs'])
    m.generate_basis(case['Na'], case['Nb'], case['Norbs'])
    return m.o2_matrix(m.basis)


def _count_ops(matrix):
    return sum(int(matrix[i, j].count_ops())
               for i in range(matrix.rows) for j in range(matrix.cols))


def _trial(method, case):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    H2 = _build_matrix_o2(method, case)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ops = _count_ops(H2)
    n = H2.rows
    return dict(wall=wall, peak_mb=peak / (1024 * 1024), ops=ops, dim=n)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float('nan')
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def benchmark():
    print('=' * 78)
    print('Two-electron matrix benchmark: direct vs blocked')
    print('=' * 78)
    print()
    header = ('| %-15s | %-7s | %-7s | dim   | wall (s)   | peak (MB)  '
              '| ops      | speedup |')
    sep = ('|-----------------|---------|---------|-------'
           '|------------|------------|----------|---------|')
    print(header % ('system', 'cutoff', 'method'))
    print(sep)
    for case, n_trials in CASES:
        cutoff = case['kwargs'].get('max_2e_centers', 4)
        results = {}
        for method in ('direct', 'blocked'):
            print('  ... %s cutoff=%d %s (n=%d)'
                  % (case['label'], cutoff, method, n_trials), flush=True)
            trials = []
            for ti in range(n_trials):
                t = _trial(method, case)
                trials.append(t)
                print('      trial %d: wall=%.3fs peak=%.1fMB'
                      % (ti, t['wall'], t['peak_mb']), flush=True)
            results[method] = dict(
                wall=_median([t['wall'] for t in trials]),
                peak_mb=_median([t['peak_mb'] for t in trials]),
                ops=trials[0]['ops'],
                dim=trials[0]['dim'],
            )
        # Print rows; speedup column shown on the blocked row only.
        for method in ('direct', 'blocked'):
            r = results[method]
            speedup = ''
            if method == 'blocked' and results['direct']['wall'] > 0:
                speedup = '%.2fx' % (results['direct']['wall'] / r['wall'])
            print('| %-15s | %-7d | %-7s | %-5d | %-10.3f | %-10.2f '
                  '| %-8d | %-7s |'
                  % (case['label'], cutoff, method, r['dim'],
                     r['wall'], r['peak_mb'], r['ops'], speedup))
        # Sanity: ops should agree (modulo chemist-symmetry symbol naming).
        ops_diff = results['direct']['ops'] - results['blocked']['ops']
        if ops_diff != 0:
            print('|   ^ ops differ by %+d (chemist-symbol naming, not '
                  'a correctness issue)' % ops_diff)
    print()
    print('Notes:')
    print(' - wall and peak_mb are medians across N_TRIALS runs.')
    print(' - ops = sum of sympy.count_ops() over all matrix entries.')
    print(' - speedup > 1 means blocked is faster.')


if __name__ == '__main__':
    benchmark()
