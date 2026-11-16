"""Combine ACIC and N&W distance--ranking results for the main text."""
from pathlib import Path

import pandas as pd

from distance_across_simulators import CAPTION_NOTE, METRICS, MODES, cells
from add_setting_win_counts import winners

TABLES = Path(__file__).resolve().parents[1] / 'tables'


def main():
    acic = pd.read_csv(TABLES / 'distance_across_simulators_acic.csv')
    nie = pd.read_csv(TABLES / 'distance_across_simulators_nie.csv')
    assert acic.n_reference_datasets.eq(100).all() and nie.n_reference_datasets.eq(100).all()
    end = r' \\'
    settings, headers = [], []
    for setting, header in (
            ('IID', 'IID'), ('Non-additive', r'\shortstack{Non-\\additive}'),
            ('Group-correlated', r'\shortstack{Group-\\correlated}'),
            ('Heteroskedastic', r'\shortstack{Hetero-\\skedastic}')):
        settings.append(acic.loc[acic.setting.eq(setting)])
        headers.append(header)
    for setting in 'ABCD':
        for n in (1000, 4302):
            settings.append(nie.loc[nie.setting.eq(setting) & nie.n.eq(n)])
            headers.append(rf'\shortstack{{{setting}\\$n={n}$}}')
    assert all(len(group) == 6 for group in settings)
    values = [cells(group) for group in settings]
    for index, group in enumerate(settings):
        indexed = group.set_index(['ranking_measure', 'distance_metric'])
        for mode_index, mode in enumerate(MODES):
            raw = [[indexed.loc[(mode, metric), 'mean_pearson'] for metric in METRICS]]
            for metric_index, won in enumerate(winners(raw)):
                if won:
                    position = 3 * mode_index + metric_index
                    values[index][position] = r'\textbf{' + values[index][position] + '}'
    for index, group in enumerate(settings):
        indexed = group.set_index(['ranking_measure', 'distance_metric'])
        for mode_index, mode in enumerate(MODES):
            for metric_index, metric in enumerate(METRICS):
                row = indexed.loc[(mode, metric)]
                percentage = 100 * row.fraction_negative
                assert abs(percentage - round(percentage)) < 1e-8
                symbol = (r'\S' if percentage >= 99 - 1e-8 else
                          r'\ddagger' if percentage >= 95 - 1e-8 else
                          r'\dagger' if percentage >= 90 - 1e-8 else '')
                if symbol:
                    values[index][3 * mode_index + metric_index] += rf'\textsuperscript{{\({symbol}\)}}'
    counts = {}
    for mode in ('full_rho', 'excl_rho'):
        raw = [[group.set_index(['ranking_measure', 'distance_metric']).loc[
            (mode, metric), 'mean_pearson'] for metric in METRICS] for group in settings]
        counts[mode] = winners(raw)
    count_note = (r' Bold marks the most negative mean correlation within each setting and estimator panel, including ties based on unrounded means. Superscripts \(\dagger\), \(\ddagger\), and \(\S\) indicate negative correlations in at least 90\%, 95\%, and 99\% of reference datasets, respectively (do not confuse with significance levels). '
                  )
    lines = [r'\begin{table*}[!t]', r'\centering',
             r'\caption{' + CAPTION_NOTE.replace('Each correlation compares the five simulators', 'The main panels compare all five simulators; sensitivity panels remove WGAN in ACIC and RF in N\\&W, leaving four. Each correlation compares these simulators') + count_note + '}',
             r'\label{tab:distance-ranking}', r'{',
             r'\makebox[\textwidth][c]{\begin{tabular}{lrrrr@{\hspace{\tablegroupsep}}rrrrrrrr@{\hspace{\tablegroupsep}}r}', r'\toprule',
             r'& \multicolumn{4}{c}{ACIC 2017} & \multicolumn{8}{c}{N\&W} & Winner count' + end,
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-13}\cmidrule(lr){14-14}',
             'Distance & ' + ' & '.join(headers) + r' & Total' + end, r'\midrule']
    for mode, title in enumerate(('All estimators', 'Excl. aligned')):
        if mode:
            lines.append(r'\addlinespace[\tablepanelgap]')
        lines.append(r'\multicolumn{14}{l}{\textit{' + title + '}}' + end)
        for metric, label in enumerate((r'$W_1$', 'KS', 'Energy')):
            key = ('full_rho', 'excl_rho')[mode]
            wins = [f'{counts[key][metric]}/{len(settings)}']
            lines.append(' & '.join([label, *[row[3 * mode + metric] for row in values], *wins]) + end)
    root = TABLES.parents[2]
    acic_drop = pd.read_csv(root / 'review/distance_without_wgan/summary.csv').rename(
        columns={'without_wgan': 'reduced_set'})
    nie_drop = pd.read_csv(root / 'review/distance_without_rf/summary.csv')
    sensitivity = pd.concat([acic_drop.loc[acic_drop.design.eq('ACIC 2017')], nie_drop])
    assert len(sensitivity) == 72
    for mode, title in zip(MODES, ('All estimators; WGAN / RF removed',
                                  'Excl. aligned; WGAN / RF removed')):
        lines.append(r'\addlinespace[\tablepanelgap]')
        lines.append(r'\multicolumn{14}{l}{\textit{' + title + '}}' + end)
        raw = [[sensitivity.loc[(sensitivity.setting_index == index)
                               & sensitivity.ranking_measure.eq(mode)
                               & sensitivity.distance_metric.eq(metric), 'reduced_set'].item()
                for metric in METRICS] for index in range(12)]
        wins = winners(raw)
        for metric_index, (metric, label) in enumerate(zip(METRICS, (r'$W_1$', 'KS', 'Energy'))):
            row_cells = []
            for index in range(12):
                row = sensitivity.loc[(sensitivity.setting_index == index)
                                      & sensitivity.ranking_measure.eq(mode)
                                      & sensitivity.distance_metric.eq(metric)].iloc[0]
                value = f'{row.reduced_set:.3f}'
                if winners([raw[index]])[metric_index]:
                    value = r'\textbf{' + value + '}'
                fraction = row.negative_without
                symbol = (r'\S' if fraction >= .99 - 1e-10 else
                          r'\ddagger' if fraction >= .95 - 1e-10 else
                          r'\dagger' if fraction >= .90 - 1e-10 else '')
                if symbol:
                    value += rf'\textsuperscript{{\({symbol}\)}}'
                row_cells.append(value)
            lines.append(' & '.join([label, *row_cells, f'{wins[metric_index]}/12']) + end)
    lines += [r'\bottomrule', r'\end{tabular}}', '}', r'\end{table*}', '']
    (TABLES / 'results_distance_rho_combined.tex').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
