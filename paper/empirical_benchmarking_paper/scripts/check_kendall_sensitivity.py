"""Reproduce the discussion's Spearman-versus-Kendall sensitivity check."""
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from paper_results_config import SETTINGS


def main():
    output = Path(__file__).resolve().parents[1] / 'exploratory/kendall_sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (design, setting, n, directory) in enumerate(SETTINGS):
        data = pd.read_csv(directory / 'promotion_detail.csv')
        for (seed, simulator), group in data.groupby(['seed', 'generator']):
            for mode in ('All', 'Excl'):
                values = group if mode == 'All' else group.loc[
                    ~group.family_aligned.astype(str).str.lower().eq('true')]
                rows.append({'setting_index': index, 'design': design, 'setting': setting,
                             'sample_size': n, 'seed': seed, 'simulator': simulator, 'mode': mode,
                             'spearman': spearmanr(values.real_rank, values.sim_rank).statistic,
                             'kendall': kendalltau(values.real_rank, values.sim_rank).statistic})
    detail = pd.DataFrame(rows)
    detail.to_csv(output / 'by_seed.csv', index=False)
    keys = ['setting_index', 'design', 'setting', 'sample_size', 'simulator', 'mode']
    summary = detail.groupby(keys, sort=False)[['spearman', 'kendall']].mean().reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    comparisons = []
    for (index, mode), group in summary.groupby(['setting_index', 'mode']):
        indexed = group.set_index('simulator')
        comparisons.append({'setting_index': index, 'mode': mode,
                            'spearman_best': indexed.spearman.idxmax(),
                            'kendall_best': indexed.kendall.idxmax(),
                            'simulator_order_spearman': spearmanr(indexed.spearman, indexed.kendall).statistic})
    pd.DataFrame(comparisons).to_csv(output / 'ordering_comparisons.csv', index=False)
    print(f'Wrote Kendall sensitivity results to {output}')


if __name__ == '__main__':
    main()
