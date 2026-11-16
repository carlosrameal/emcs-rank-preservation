"""Combined ACIC/N&W appendix tables, with landscape longtable pagination."""
import json
from pathlib import Path

from paper_results_config import SETTINGS
from paper_results_data import _read_mse, _read_rho, DISPLAY_NAMES, promotion_by_type

TABLES = Path(__file__).resolve().parents[1] / 'tables'
END = r' \\'


def start_table(spec, caption, label, columns, headers):
    lines = [r'\begin{longtable}{' + spec + '}', r'\caption{' + caption + r'}\label{' + label + '}' + END,
             r'\toprule', *headers, r'\midrule', r'\endfirsthead',
             r'\caption[]{' + caption.split('.')[0] + ' (continued).}' + END,
             r'\toprule', *headers, r'\midrule', r'\endhead',
             r'\midrule', rf'\multicolumn{{{columns}}}{{r}}{{Continued on next page}}' + END,
             r'\endfoot', r'\bottomrule', r'\endlastfoot']
    return lines


def rank_values(index, simulator):
    directory = SETTINGS[index][3]
    rho = _read_rho(directory).set_index('generator')
    movement = promotion_by_type((directory,))
    return [f'{rho.loc[simulator, "avg_full_rho"]:.3f}', f'{rho.loc[simulator, "avg_excl_rho"]:.3f}',
            *[f'{movement.loc[simulator, metric + "_" + kind]:.2f}'
              for metric in ('promotion', 'shift') for kind in ('gcomp', 'aipw', 'tmle', 'rest')]]


def write_ranks():
    group = (r'\multicolumn{2}{c}{Rank correlation} & \multicolumn{4}{c}{Mean promotion} & '
             r'\multicolumn{4}{c}{Mean abs. shift}')
    names = 'All & Excl. & G-comp & AIPW & TMLE & Rest & G-comp & AIPW & TMLE & Rest'
    headers = [' & ' + group + ' & ' + group + END,
               r'\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-13}\cmidrule(lr){14-17}\cmidrule(lr){18-21}',
               'Simulator & ' + names + ' & ' + names + END]
    caption = ('Ranking fidelity and rank movement for all ACIC 2017 and N\\&W settings, '
               'computed over 100 reference datasets per setting. All uses all 15 estimators; Excl. removes '
               'estimators sharing the simulator outcome-model family. Mean promotion and mean absolute rank '
               'shift use positions in the full rankings and are reported for the associated G-comp, aligned '
               'AIPW, aligned TMLE, and non-aligned estimators (Rest). Positive promotion indicates a higher '
               'rank under the simulator. OLS1 is aligned with OLS2 and excluded from Rest.')
    lines = start_table(r'l*{10}{r}@{\hspace{\tablegroupsep}}*{10}{r}', caption,
                        'tab:results_rho_all', 21, headers)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    count = 0
    for block, pair in enumerate(pairs):
        if block:
            lines.append(r'\midrule')
        titles = []
        for index in pair:
            design, setting, n, _ = SETTINGS[index]
            title = f'ACIC {setting}' if design == 'ACIC 2017' else rf'N\&W {setting}, $n={n}$'
            titles.append(rf'\multicolumn{{10}}{{c}}{{{title}}}')
        lines += [' & ' + ' & '.join(titles) + END + '*', r'\cmidrule(lr){2-11}\cmidrule(lr){12-21}']
        for row, simulator in enumerate(('GB', 'NN', 'OLS', 'RF')):
            values = [value for index in pair for value in rank_values(index, simulator)]
            lines.append(' & '.join([{'OLS': 'OLS2'}.get(simulator, simulator), *values]) + END + ('*' if row < 3 else ''))
            count += len(values)
    lines += [r'\end{longtable}', '']
    (TABLES / 'results_rho_all.tex').write_text('\n'.join(lines), encoding='utf-8')
    assert count == 480
    return count


def write_mse():
    """Two-page table with independently ordered setting blocks."""
    caption = ('Average Monte Carlo MSE by estimator and benchmark, averaged over 100 reference '
               'datasets per setting. Each setting has its own estimator order, from lowest to highest target-benchmark MSE.')
    block_header = [r'\makebox[54pt][c]{Estimator}'] + [
        r'\makebox[19pt][c]{' + name + '}' for name in ['Target', 'GB', 'NN', 'OLS2', 'RF', 'WGAN']]
    header = ' & '.join(block_header * 4) + END
    lines = [r'\begingroup',
             r'\renewcommand{\papertablestyle}{\fontsize{7.5}{10}\selectfont\setlength{\tabcolsep}{0.5pt}\renewcommand{\arraystretch}{1}}']
    lines += start_table(r'*{7}{c}@{\hspace{6pt}}*{7}{c}@{\hspace{6pt}}*{7}{c}@{\hspace{6pt}}*{7}{c}', caption,
                         'tab:results_mse_all', 28, [header])
    count = 0
    benchmarks = ['REAL_DGP', 'GB', 'NN', 'OLS', 'RF', 'WGAN']
    for index in range(0, 12, 4):
        if index == 8:
            lines.append(r'\pagebreak')
        elif index:
            lines.append(r'\addlinespace[14pt]')
        frames, titles = [], []
        for setting_index in range(index, index + 4):
            design, setting, n, directory = SETTINGS[setting_index]
            frame = _read_mse(directory).sort_values('REAL_DGP', kind='stable')
            assert frame.REAL_DGP.is_monotonic_increasing
            frames.append(frame)
            title = f'ACIC {setting}' if design == 'ACIC 2017' else rf'N\&W {setting}, $n={n}$'
            titles.append(r'\multicolumn{7}{c}{\textit{' + title + '}}')
        lines += [' & '.join(titles) + END + '*', r'\cmidrule(lr){1-7}\cmidrule(lr){8-14}\cmidrule(lr){15-21}\cmidrule(lr){22-28}']
        for row_index in range(15):
            cells = []
            for frame in frames:
                row = frame.iloc[row_index]
                cells += [r'\makebox[54pt][c]{' + DISPLAY_NAMES[frame.index[row_index]] + '}',
                          *[r'\makebox[19pt][c]{' + f'{float(row[b]):.3f}' + '}' for b in benchmarks]]
                count += len(benchmarks)
            lines.append(' & '.join(cells) + END + ('*' if row_index < 14 else ''))
    lines += [r'\end{longtable}', r'\endgroup', '']
    (TABLES / 'results_mse_all.tex').write_text('\n'.join(lines), encoding='utf-8')
    assert count == 1080
    return count


def main():
    report = {'mse_values': write_mse(), 'ranking_and_movement_values': write_ranks(),
              'settings': 12, 'reference_datasets_per_setting': 100,
              'mse_estimator_order': 'setting-specific target MSE',
              'pagination': 'two-page MSE longtable; landscape ranking longtable'}
    (TABLES / 'combined_appendix_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
