"""Plot setting-specific rank promotion against absolute rank movement."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from paper_results_data import FAMILIES, SIMULATORS, DISPLAY, promotion_by_type

OUT = Path(__file__).resolve().parents[1] / 'figures' / 'promotion_shift'
# Okabe-Ito categorical colors; avoid the low-contrast yellow on white.
COLORS = dict(zip(SIMULATORS, ['#0072B2', '#D55E00', '#009E73', '#CC79A7']))
MARKERS = dict(zip(SIMULATORS, ['o', '^', 's', 'D']))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, directories in FAMILIES.items():
        for directory in directories:
            summary = promotion_by_type((directory,))
            for simulator, row in summary.iterrows():
                rows.append(dict(design=family, setting=directory.name,
                                 simulator=simulator, **row.to_dict()))
    data = pd.DataFrame(rows)
    data.to_csv(OUT / 'promotion_shift_points.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'font.family': 'serif',
                         'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#999999', 'axes.linewidth': .6,
                         'xtick.color': '#555555', 'ytick.color': '#555555',
                         'xtick.major.size': 3, 'ytick.major.size': 3,
                         'xtick.major.width': .6, 'ytick.major.width': .6,
                         'pdf.fonttype': 42, 'mathtext.fontset': 'stix'})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    for panel, (ax, kind, title) in enumerate(zip(axes.flat, ['gcomp', 'aipw', 'tmle', 'rest'],
                               ['G-comp', 'AIPW', 'TMLE', 'Rest'])):
        ax.plot([-15, 0, 15], [15, 0, 15], color='#c5c5c5', lw=.7,
                ls=(0, (4, 4)), zorder=0)
        ax.axvline(0, color='#d4d4d4', lw=.6, zorder=0)
        for simulator in SIMULATORS:
            for acic in [False, True]:
                group = data.loc[data.simulator.eq(simulator) &
                                 data.design.eq('ACIC 2017').eq(acic)]
                ax.scatter(group['promotion_' + kind], group['shift_' + kind],
                           marker=MARKERS[simulator], s=40 if acic else 34,
                           facecolors='none' if acic else COLORS[simulator],
                           edgecolors=COLORS[simulator], linewidths=1.05 if acic else .5,
                           alpha=1, zorder=3 if acic else 2)
        ax.set_title(f'({chr(97 + panel)})  {title}', loc='left', fontsize=12, pad=10)
        ax.set(xlim=(-7.5, 14.7), ylim=(-.35, 14.7),
               xticks=[-5, 0, 5, 10], yticks=[0, 5, 10])
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#eeeeee', linewidth=.6)
    fig.supxlabel('Mean rank promotion', y=.105, fontsize=11)
    fig.supylabel('Mean absolute rank shift', x=.025, fontsize=11)
    handles = [Line2D([], [], color=COLORS[s], marker=MARKERS[s], ls='',
                      markersize=6, label=DISPLAY.get(s, s))
               for s in SIMULATORS]
    handles += [Line2D([], [], color='#666666', marker='o', mfc='none', ls='', label='ACIC'),
                Line2D([], [], color='#777777', marker='o', ls='', label='N&W', alpha=1)]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
               handlelength=1.2, columnspacing=1.8, bbox_to_anchor=(.53, .018))
    fig.subplots_adjust(left=.105, right=.985, bottom=.215, top=.94,
                        wspace=.15, hspace=.28)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / ('promotion_shift.' + ext), dpi=220)
    plt.close(fig)

if __name__ == '__main__':
    main()
