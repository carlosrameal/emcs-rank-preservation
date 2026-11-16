"""Compare competing simulators within each reference dataset, then average."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SIMULATORS = ('GB', 'NN', 'OLS', 'RF', 'WGAN')
METRICS = ('W1_mean', 'KS_mean', 'Energy_mean')
MODES = ('full_rho', 'excl_rho')


def correlations(directory: Path):
    distances = pd.read_csv(directory / 'distance_run_level.csv')
    if 'group' in distances:
        assert distances.group.eq('ALL').all()
    ranks = pd.read_csv(directory / 'per_seed_rho.csv')
    data = distances.drop(columns=list(MODES), errors='ignore').merge(
        ranks[['seed', 'generator', *MODES]], on=['seed', 'generator'], validate='one_to_one')
    assert len(data) == 500 and set(data.seed) == set(range(2001, 2101))
    assert np.isfinite(data[[*METRICS, *MODES]].to_numpy()).all()
    rows = []
    for seed, group in data.groupby('seed', sort=True):
        assert len(group) == 5 and set(group.generator) == set(SIMULATORS)
        for mode in MODES:
            for metric in METRICS:
                x, y = group[metric].to_numpy(), group[mode].to_numpy()
                if np.ptp(x) == 0 or np.ptp(y) == 0:
                    raise ValueError(f'Undefined within-reference correlation: {directory}, {seed}, {mode}, {metric}')
                coefficient = float(pearsonr(x, y).statistic)
                # Independent centered-vector check on the Pearson calculation.
                xc, yc = x - x.mean(), y - y.mean()
                np.testing.assert_allclose(coefficient, xc @ yc / np.sqrt((xc @ xc) * (yc @ yc)), atol=1e-12)
                rows.append({'seed': seed, 'ranking_measure': mode, 'distance_metric': metric,
                             'pearson': coefficient, 'spearman': float(spearmanr(x, y).statistic),
                             'n_simulators': len(group)})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(['ranking_measure', 'distance_metric'], sort=False).agg(
        mean_pearson=('pearson', 'mean'), sd_pearson=('pearson', 'std'),
        median_pearson=('pearson', 'median'), mean_spearman=('spearman', 'mean'),
        fraction_negative=('pearson', lambda x: float((x < 0).mean())),
        n_reference_datasets=('seed', 'nunique')).reset_index()
    assert summary.n_reference_datasets.eq(100).all()
    return summary, detail


CAPTION_NOTE = (
    'Mean within-reference-dataset Pearson correlations between outcome-distribution distance '
    'and Spearman ranking fidelity. Each correlation compares the five simulators on the same '
    'reference dataset; coefficients are then averaged over the 100 reference datasets. '
    'All and Excl. aligned use ranking fidelity before and after excluding estimators aligned '
    'with each simulator, respectively. Negative values indicate that '
    'simulators with smaller distances tend to have higher ranking fidelity.')


def cells(summary):
    values = summary.set_index(['ranking_measure', 'distance_metric']).mean_pearson
    return [f'{values.loc[(mode, metric)]:.3f}' for mode in MODES for metric in METRICS]


def render(panels, *, caption, label, with_n=False):
    leading = 2 if with_n else 1
    end = r' \\'
    lines = [r'\begin{table}[!htbp]', r'\centering', r'\caption{' + caption + '}',
             r'\label{' + label + '}', r'{', r'\begin{tabular}{' + 'l' * leading + 'rrrrrr}',
             r'\toprule', ' & ' * leading + r'\multicolumn{3}{c}{All estimators} & \multicolumn{3}{c}{Excl. aligned}' + end,
             f'\\cmidrule(lr){{{leading+1}-{leading+3}}}\\cmidrule(lr){{{leading+4}-{leading+6}}}',
             ('DGP & $n$' if with_n else 'Error structure') + r' & $W_1$ & KS & Energy & $W_1$ & KS & Energy' + end,
             r'\midrule']
    for setting, n, data in panels:
        prefix = [setting, str(n)] if with_n else [setting]
        lines.append(' & '.join(prefix + cells(data)) + end)
    lines.extend([r'\bottomrule', r'\end{tabular}', '}', r'\end{table}', ''])
    return '\n'.join(lines)
