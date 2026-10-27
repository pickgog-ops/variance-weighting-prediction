"""Entry point for the main experiments and the variance-shrinkage rules.

Each stage runs in its own work directory, so the files in data/, results/ and
reference/ are never changed.

  python reproduce.py prepare --stage all --work-dir runs/replay --saved
  python reproduce.py run --stage main --action analyze --work-dir runs/replay
  python reproduce.py run --stage shrinkage_rules --action analyze --work-dir runs/replay

Stages and actions
  main             analyze  paired tests and main-text table values from the saved fold-level results
                   full     rerun all model fits, then analyze (slow)
  shrinkage_rules  analyze  summaries and paired tests from the saved records
                   tables   LaTeX tables
                   full     rerun the simulation and the benchmark comparison, then analyze (slow)
"""
from pathlib import Path
import argparse, os, shutil, subprocess, sys

PACKAGE = Path(__file__).resolve().parent
STAGES = ('main', 'shrinkage_rules')


def copytree(source, dest):
    shutil.copytree(source, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def prepare(stage, work, saved):
    root = work / stage
    if root.exists():
        raise FileExistsError(f'{root} already exists; choose a new --work-dir')
    root.mkdir(parents=True)
    (root / '.wls_stage.json').write_text('{"stage": "%s"}' % stage, encoding='utf-8')
    copytree(PACKAGE / 'code' / stage, root / 'code')
    copytree(PACKAGE / 'data', root / 'data')
    (root / 'results').mkdir()
    if saved:
        copytree(PACKAGE / 'results' / stage, root / 'results')
    copytree(PACKAGE / 'reference', root / 'reference')
    return root


def commands(stage, action, root):
    if stage == 'main':
        analyze = [['benchmark_tests.py'], ['main_text_tables.py']]
        if action == 'analyze': return analyze
        if action == 'full':
            return [[x] for x in ['run_experiments.py', 'finalize_results.py', 'exp_scoring_rules.py',
                                  'exp_simulation.py', 'std_penalized.py']] + analyze
    if stage == 'shrinkage_rules':
        if action == 'analyze': return [['analyze_c123.py']]
        if action == 'tables':
            (root / 'tables').mkdir(exist_ok=True)
            return [['make_c123_tables.py', str(root / 'results'), str(root / 'tables')]]
        if action == 'full':
            return [['run_simulation.py', '500', '200', '2000'], ['run_realdata.py'], ['analyze_c123.py']]
    raise ValueError(f'unknown action {action!r} for stage {stage!r}; see README.md')


def run_child(root, script, args=()):
    env = os.environ.copy()
    env.update({'WLS_STAGE_ROOT': str(root), 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8',
                'PYTHONDONTWRITEBYTECODE': '1', 'MPLBACKEND': 'Agg'})
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    return subprocess.run([sys.executable, '-X', 'utf8', str(script), *args], cwd=root, env=env).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['prepare', 'run'])
    ap.add_argument('--stage', choices=['all', *STAGES], default='all')
    ap.add_argument('--action', default='analyze')
    ap.add_argument('--work-dir', type=Path, default=PACKAGE / 'runs' / 'replay')
    ap.add_argument('--saved', action='store_true', help='copy the saved results for an analysis-only replay')
    args = ap.parse_args()
    work = args.work_dir.resolve()
    for protected in ('data', 'results', 'reference', 'code'):
        p = PACKAGE / protected
        if work == p or work.is_relative_to(p) or p.is_relative_to(work):
            sys.exit('--work-dir must be outside the repository folders data/, results/, reference/ and code/')
    if args.command == 'prepare':
        for stage in (STAGES if args.stage == 'all' else [args.stage]):
            print('Prepared', prepare(stage, work, args.saved))
        return 0
    if args.stage == 'all':
        sys.exit('run requires one --stage')
    root = work / args.stage
    if not (root / '.wls_stage.json').exists():
        sys.exit('Run "prepare" first')
    for cmd in commands(args.stage, args.action, root):
        code = run_child(root, root / 'code' / cmd[0], cmd[1:])
        if code:
            return code
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
