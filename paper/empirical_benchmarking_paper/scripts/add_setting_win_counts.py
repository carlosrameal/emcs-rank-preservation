"""Add setting-level win summaries to main-text Tables 2--4."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
TABLES = PAPER / 'tables'
END = r' \\'


def winners(values, maximize=False):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    best = a.max(axis=1) if maximize else a.min(axis=1)
    return np.isclose(a, best[:, None], atol=1e-12, rtol=0).sum(axis=0)


def update(name, rows, note=''):
    path = TABLES / name
    text = path.read_text(encoding='utf-8')
    text = re.sub(r'% BEGIN WIN COUNTS.*?% END WIN COUNTS\n', '', text, flags=re.S)
    # Keep generation idempotent, including caption additions.
    text = re.sub(r' Counts: .*?(?=\}\n\\label)', '', text, flags=re.S)
    pos = text.index('}\n\\label')
    text = text[:pos] + (' Counts: ' + note if note else '') + text[pos:]
    block = '% BEGIN WIN COUNTS\n' + '\n'.join(rows) + '\n% END WIN COUNTS\n'
    text = text.replace(r'\bottomrule', block + r'\bottomrule')
    path.write_text(text, encoding='utf-8')


def rank_counts():
    frame = pd.read_csv(TABLES / 'rank_aggregation_summary.csv')
    sims = ['GB', 'NN', 'OLS', 'RF', 'WGAN', 'aggregate']
    rows = [r'\midrule']
    all_wins = winners(frame[[s + '_all' for s in sims]], True)
    excl_wins = winners(frame[[s + '_excl' for s in sims]], True)
    gap_wins = winners(frame[[s + '_abs_gap' for s in sims[:5]]])
    cells = []
    for i in range(5):
        cells.append(f'{all_wins[i]}/{len(frame)}')
        if i != 4:
            cells.append(f'{excl_wins[i]}/{len(frame)}')
        cells.append(f'{gap_wins[i]}/{len(frame)}')
    cells += [f'{all_wins[5]}/{len(frame)}', f'{excl_wins[5]}/{len(frame)}']
    rows.append('Winner count & ' + ' & '.join(cells) + END)
    update('results_rank_aggregation.tex', rows,
           'wins out of 12 settings, based on unrounded values.')


def regret_counts():
    frame = pd.read_csv(TABLES / 'gcomp_selection_summary.csv')
    rows = [r'\midrule']
    indices = sorted(frame.setting_index.unique())
    cells = []
    for sim in ['GB', 'NN', 'OLS', 'RF']:
        values = []
        for index in indices:
            group = frame.loc[frame.setting_index.eq(index) & frame.generator.eq(sim)].set_index('selection')
            values.append([group.loc['All', 'mean_gcomp_relative_regret_pct'],
                           group.loc['All', 'mean_selected_relative_regret_pct'],
                           group.loc['Excl', 'mean_selected_relative_regret_pct']])
        cells += [f'{x}/{len(indices)}' for x in winners(values)]
    rows.append('Winner count & ' + ' & '.join(cells) + END)
    update('results_gcomp_selection.tex', rows)


def distance_counts():
    from make_distance_table import main
    main()


if __name__ == '__main__':
    rank_counts()
    regret_counts()
    distance_counts()
