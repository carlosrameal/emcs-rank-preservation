"""Sensitivity of within-reference distance correlations to removing WGAN."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper/empirical_benchmarking_paper/scripts'))
from paper_results_config import SETTINGS
from distance_across_simulators import correlations, METRICS, MODES

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exclude', choices=['WGAN', 'RF'], default='WGAN')
args = parser.parse_args()
excluded = args.exclude
OUT = ROOT / ('review/distance_without_' + excluded.lower())
OUT.mkdir(exist_ok=True, parents=True)
records, details = [], []
for index, (design, setting, n, directory) in enumerate(SETTINGS):
    if excluded == 'RF' and design != 'Nie and Wager':
        continue
    _, full = correlations(directory)
    distances = pd.read_csv(directory / 'distance_run_level.csv')
    ranks = pd.read_csv(directory / 'per_seed_rho.csv')
    data = distances.drop(columns=list(MODES), errors='ignore').merge(
        ranks[['seed', 'generator', *MODES]], on=['seed', 'generator'], validate='one_to_one')
    data = data.loc[data.generator.ne(excluded)]
    assert len(data) == 400
    rows = []
    for seed, group in data.groupby('seed'):
        assert set(group.generator) == {'GB', 'NN', 'OLS', 'RF', 'WGAN'} - {excluded}
        for mode in MODES:
            for metric in METRICS:
                x, y = group[metric].to_numpy(), group[mode].to_numpy()
                assert np.ptp(x) > 0 and np.ptp(y) > 0
                rows.append(dict(seed=seed, ranking_measure=mode, distance_metric=metric,
                                 reduced_set=float(pearsonr(x, y).statistic)))
    paired = pd.DataFrame(rows).merge(full[['seed','ranking_measure','distance_metric','pearson']],
                                     on=['seed','ranking_measure','distance_metric'], validate='one_to_one')
    paired = paired.rename(columns={'pearson':'full_set'})
    paired['change'] = paired.reduced_set - paired.full_set
    paired = paired.assign(setting_index=index, design=design, setting=setting, n=n)
    details.append(paired)
    for (mode, metric), group in paired.groupby(['ranking_measure','distance_metric'],sort=False):
        records.append(dict(setting_index=index, design=design, setting=setting, n=n,
                            ranking_measure=mode, distance_metric=metric,
                            full_set=group.full_set.mean(), reduced_set=group.reduced_set.mean(),
                            change=group.change.mean(),
                            negative_with=(group.full_set<0).mean(),
                            negative_without=(group.reduced_set<0).mean(),
                            sd_without=group.reduced_set.std(), n_reference=len(group)))
result = pd.DataFrame(records)
pd.concat(details).to_csv(OUT/'paired_correlations.csv',index=False)
result.to_csv(OUT/'summary.csv',index=False)
assert len(result)==(72 if excluded == 'WGAN' else 48) and result.n_reference.eq(100).all()
for mode, group in result.groupby('ranking_measure'):
    print(mode, 'negative means:',int((group.full_set<0).sum()),'->',int((group.reduced_set<0).sum()),
          'mean abs change:',round(group.change.abs().mean(),3),
          'max abs change:',round(group.change.abs().max(),3))
print(result[['design','setting','n','ranking_measure','distance_metric','full_set','reduced_set','change']].to_string(index=False,float_format=lambda x:f'{x:.3f}'))
