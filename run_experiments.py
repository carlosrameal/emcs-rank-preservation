"""Portable launcher for the twelve manuscript settings or a small demo."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    settings = ['acic-iid', 'acic-nonadditive', 'acic-group_corr', 'acic-heteroskedastic']
    settings += [f'nie-{design}-{n}' for design in 'ABCD' for n in (1000, 4302)]
    parser.add_argument('--setting', choices=settings)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--seed', type=int, default=2001)
    parser.add_argument('--all-seeds', action='store_true', help='Run seeds 2001 through 2100 sequentially.')
    parser.add_argument('--acic-data', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / '_build/experiments')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.demo and args.setting is None:
        parser.error('Choose --demo or --setting.')
    if args.demo and args.all_seeds:
        parser.error('The demo runs a single reference dataset.')
    setting = 'demo' if args.demo else args.setting
    for seed in (range(2001, 2101) if args.all_seeds else [args.seed]):
        output = args.output.resolve() / setting / f'seed{seed}'
        command = [sys.executable, str(ROOT / 'mc_study_main.py'), '--job-seed', str(seed),
                   '--n-mc', '5' if args.demo else '1000', '--n-workers', str(args.workers),
                   '--threads-per-worker', '1', '--full-output', '0', '--output-root', str(output)]
        if args.demo:
            command += ['--dgp', 'NieA', '--synthetic-n', '100', '--synthetic-d', '6',
                        '--generators', 'OLS', '--estimators', 'OLS1', 'OLS2_GCOMP']
        elif setting.startswith('nie-'):
            _, design, n = setting.split('-')
            command += ['--dgp', 'Nie' + design, '--synthetic-n', n,
                        '--synthetic-d', '6', '--synthetic-sigma', '1']
        else:
            if args.acic_data is None:
                parser.error('ACIC experiments require --acic-data PATH.')
            command += ['--data-path', str(args.acic_data.resolve()), '--acic-error', setting[5:],
                        '--acic-treatment-mode', 'redraw', '--acic-magnitude', '1',
                        '--acic-noise', '1', '--acic-confounding', '1']
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            if output.exists():
                raise FileExistsError(f'Choose a new output directory: {output}')
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
