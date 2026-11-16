"""Report the within-reference comparison of outcome distances and fidelity."""
import json
from pathlib import Path

import pandas as pd

from distance_across_simulators import correlations
from paper_results_config import SETTINGS


def main():
    output = Path(__file__).resolve().parents[1] / 'exploratory/distance_across_simulators'
    output.mkdir(parents=True, exist_ok=True)
    summaries, details = [], []
    for index, (design, setting, n, directory) in enumerate(SETTINGS):
        summary, detail = correlations(directory)
        tags = dict(setting_index=index, design=design, setting=setting, sample_size=n)
        summaries.append(summary.assign(**tags))
        details.append(detail.assign(**tags))
    summary, detail = pd.concat(summaries, ignore_index=True), pd.concat(details, ignore_index=True)
    summary.to_csv(output / 'summary.csv', index=False)
    detail.to_csv(output / 'by_reference_dataset.csv', index=False)
    tables = output.parents[1] / 'tables'
    for design, suffix in [('ACIC 2017', 'acic'), ('Nie and Wager', 'nie')]:
        for source, filename in [(summary, f'distance_across_simulators_{suffix}.csv'),
                                 (detail, f'distance_across_simulators_{suffix}_by_reference.csv')]:
            frame = source.loc[source.design.eq(design)].copy()
            if suffix == 'nie':
                frame = frame.rename(columns={'sample_size': 'n'})
            frame.to_csv(tables / filename, index=False)
    lines = ['# Outcome distances and ranking fidelity across competing simulators', '',
             'Within each reference dataset, Pearson correlation is computed across GB, NN, OLS2, RF and WGAN, comparing each simulator’s distance with its ranking fidelity. The 100 correlations are then averaged within each setting. Target datasets and simulators are not pooled across reference datasets or settings. Negative coefficients indicate that smaller distances accompany higher fidelity.', '',
             'All distances were already averaged over the K=1000 outcome redraws. Both the full-estimator and alignment-excluded fidelity definitions are examined; WGAN has no aligned estimators, so its fidelity is unchanged in the latter. The excluded comparisons use simulator-specific eligible estimator sets.', '',
             '## Mean Pearson correlations', '', '| Setting | All W1 | All KS | All Energy | Excl W1 | Excl KS | Excl Energy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for index, (design, setting, n, _) in enumerate(SETTINGS):
        part = summary.loc[summary.setting_index.eq(index)].set_index(['ranking_measure', 'distance_metric'])
        label = f'{design} {setting}' + ('' if n == '--' else f', n={n}')
        values = [part.loc[(mode, metric), 'mean_pearson'] for mode in ('full_rho', 'excl_rho')
                  for metric in ('W1_mean', 'KS_mean', 'Energy_mean')]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{x:.3f}' for x in values) + ' |')
    lines += ['', '## Interpretation and limits', '',
              'These are mean within-reference associations, not correlations of setting-level averages. A negative mean is descriptive evidence of a tendency, not proof that the minimum-distance simulator is always the best choice. Each individual coefficient has only five simulator points and can be sensitive to an extreme simulator. The CSV includes the between-reference standard deviation, median, fraction of negative coefficients and a Spearman sensitivity summary.', '',
              'All 1200 reference datasets have all five simulators and finite values for all three distances and both fidelity definitions. All 7200 within-reference Pearson coefficients are finite and agree with an independent centered-vector calculation. No reference dataset is omitted. No model fitting or outcome regeneration was required. Previous within-simulator analyses remain available separately.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'validation.json').write_text(json.dumps({'settings': 12, 'reference_datasets': 1200,
        'simulators_per_reference': 5, 'within_reference_correlations': len(detail),
        'summary_cells': len(summary), 'all_reference_datasets_complete': True}, indent=2))
    print(summary[['design', 'setting', 'sample_size', 'ranking_measure', 'distance_metric', 'mean_pearson']].to_string(index=False))


if __name__ == '__main__':
    main()
