"""Replay the saved analyses and compare them with the saved results.

    python run_checks.py

No model is refitted. All outputs go to a new folder runs/check_<date-time>/;
the files in this repository are not changed. Takes about one minute.
"""
import argparse, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
REPLAYS = [('main', 'analyze'), ('shrinkage_rules', 'analyze'), ('shrinkage_rules', 'tables')]
CGLS_SCRIPTS = ['validate.py', 'analyze.py', 'analyze_extra.py', 'analyze_posthoc.py', 'feasible_target.py',
                'make_tables.py', 'make_fig.py']


def same_table(a, b):
    import numpy as np, pandas as pd
    x = pd.read_csv(a, float_precision='round_trip', low_memory=False)
    y = pd.read_csv(b, float_precision='round_trip', low_memory=False)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return False
    keys = [c for c in ('cid', 'method', 'ref', 'metric') if c in x.columns]
    if keys:  # some files are written in completion order
        x = x.sort_values(keys, kind='stable').reset_index(drop=True); y = y.sort_values(keys, kind='stable').reset_index(drop=True)
    for c in x.columns:
        if x[c].dtype.kind in 'fi' and y[c].dtype.kind in 'fi':
            if not np.allclose(x[c].astype(float), y[c].astype(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                return False
        elif not x[c].fillna('').astype(str).equals(y[c].fillna('').astype(str)):
            return False
    return True


def same_text(a, b):
    return a.read_bytes().replace(b'\r\n', b'\n') == b.read_bytes().replace(b'\r\n', b'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work-dir', help='new output folder (default: runs/check_<date-time>)')
    work = ap.parse_args().work_dir
    work = Path(work) if work else HERE / 'runs' / datetime.now().strftime('check_%Y%m%d_%H%M%S')
    if work.exists():
        sys.exit(f'{work} already exists; choose a new --work-dir')
    logs = work / 'logs'; logs.mkdir(parents=True)
    stages = work / 'stages'
    rows = []

    def step(name, cmd, cwd=HERE):
        t = time.time()
        with open(logs / (name.replace(' ', '_') + '.log'), 'w', encoding='utf-8') as out:
            code = subprocess.call(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)
        rows.append((name, 'PASS' if code == 0 else 'FAIL', f'{time.time() - t:5.1f} s'))
        print(f'{rows[-1][1]}  {name}', flush=True)
        return code == 0

    print(f'Output folder: {work}\n', flush=True)
    if step('prepare work folders', [PY, 'reproduce.py', 'prepare', '--stage', 'all', '--work-dir', str(stages), '--saved']):
        for stage, action in REPLAYS:
            step(f'replay {stage} {action}', [PY, 'reproduce.py', 'run', '--stage', stage, '--action', action,
                                             '--work-dir', str(stages)])
    cg = work / 'correlated_gls_study'
    shutil.copytree(HERE / 'correlated_gls_study', cg)
    for s in CGLS_SCRIPTS:
        step(f'correlated GLS {s}', [PY, s], cwd=cg)

    n_same, diffs = 0, []
    for stage in ('main', 'shrinkage_rules'):
        for p in sorted((HERE / 'results' / stage).glob('*.csv')):
            q = stages / stage / 'results' / p.name
            if q.exists() and same_table(p, q): n_same += 1
            else: diffs.append(f'results/{stage}/{p.name}')
    for p in sorted((HERE / 'results' / 'shrinkage_rules' / 'tables').glob('*.tex')):
        q = stages / 'shrinkage_rules' / 'tables' / p.name
        if q.exists() and same_text(p, q): n_same += 1
        else: diffs.append(f'results/shrinkage_rules/tables/{p.name}')
    for p in sorted((HERE / 'correlated_gls_study').iterdir()):
        if p.suffix in ('.csv', '.tex', '.txt'):
            q = cg / p.name
            if (same_table(p, q) if p.suffix == '.csv' else same_text(p, q)): n_same += 1
            else: diffs.append(f'correlated_gls_study/{p.name}')
    rows.append(('replayed files equal to saved files', 'PASS' if not diffs else 'FAIL',
                 f'{n_same} equal, {len(diffs)} different'))

    width = max(len(r[0]) for r in rows)
    lines = ['', 'Summary', '-' * (width + 22)] + [f'{r[1]:4}  {r[0]:<{width}}  {r[2]}' for r in rows]
    if diffs: lines += ['', 'Different files:'] + diffs
    n_fail = sum(r[1] == 'FAIL' for r in rows)
    lines += ['', 'All checks passed.' if n_fail == 0 else f'{n_fail} check(s) failed; see {logs}.']
    text = '\n'.join(lines)
    print(text)
    (work / 'check_summary.txt').write_text(text + '\n', encoding='utf-8')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
