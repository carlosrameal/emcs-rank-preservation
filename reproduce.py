"""Rebuild manuscript results from the compact numerical archive."""
from pathlib import Path
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / '_build/results'


def main():
    with zipfile.ZipFile(ROOT / 'data/final_results.zip') as z:
        performance = pd.read_csv(io.BytesIO(z.read('performance.csv'))).rename(
            columns={'evaluation_reference_value': 'ate_true'})
        distances = pd.read_csv(io.BytesIO(z.read('outcome_distances.csv')))
        settings = json.loads(z.read('settings.json'))
    assert not performance.duplicated(['setting', 'seed', 'generator', 'estimator']).any()
    assert len(performance) == 108000
    assert performance.groupby(['setting', 'seed', 'generator']).size().eq(15).all()
    assert performance.N_MC.eq(1000).all()
    np.testing.assert_allclose(performance.mc_mean_ate - performance.ate_true,
                               performance.mc_bias, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(performance.mc_mse, performance.mc_bias**2 + .999*performance.mc_var,
                               atol=1e-10, rtol=1e-10)
    (RESULTS / 'data').mkdir(parents=True, exist_ok=True)
    for name, frame in [('simulator_performance', performance.loc[performance.generator.ne('REAL_DGP')]),
                        ('target_performance', performance.loc[performance.generator.eq('REAL_DGP')]),
                        ('distribution_distances', distances)]:
        frame.to_csv(RESULTS / 'data' / (name + '.csv'), index=False)
    for setting in settings:
        name, family = setting['id'], setting['family']
        part = performance.loc[performance.setting.eq(name)]
        assert set(part.seed) == set(range(2001, 2101))
        destination = RESULTS / family / (name + '.zip')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
            for seed, frame in part.groupby('seed', sort=True):
                prefix = f'{name}_seed{seed}/'
                sim = frame.loc[frame.generator.ne('REAL_DGP')].drop(columns=['setting', 'seed'])
                target = frame.loc[frame.generator.eq('REAL_DGP')].drop(columns=['setting', 'seed'])
                dist = distances.loc[distances.setting.eq(name) & distances.seed.eq(seed)].drop(columns=['setting', 'seed'])
                assert len(sim) == 75 and len(target) == 15 and len(dist) > 0
                z.writestr(prefix + 'MASTER_mc_summary.csv', sim.to_csv(index=False))
                z.writestr(prefix + '_dgp_benchmark/target/MASTER_mc_summary.csv', target.to_csv(index=False))
                z.writestr(prefix + 'MASTER_distribution_distances_summary.csv', dist.to_csv(index=False))
    (RESULTS / 'validation.json').write_text(json.dumps({
        'complete_analysis_runs': 1200, 'pending_runs': 0,
        'analysis_seed_counts_match_manifest': True}), encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'paper/empirical_benchmarking_paper/scripts/refresh_paper_results.py')],
                   cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
