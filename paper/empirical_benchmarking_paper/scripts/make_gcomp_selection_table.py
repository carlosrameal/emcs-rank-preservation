"""Reproduce the G-comp/selection comparison from the paper's per-seed metrics.

Uses the same helper as analyze_simulator_bundle.py, which also writes these
outputs whenever a result bundle is analyzed. No rankings of averaged MSEs are
used: selection and target evaluation are paired within each reference dataset.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from paper_results_config import PROJECT_ROOT, SETTINGS

sys.path.insert(0, str(PROJECT_ROOT))
from analyze_simulator_bundle import write_gcomp_selection_comparison


TABLE_DIR = Path(__file__).resolve().parents[1] / "tables"
SIMULATORS = ("GB", "NN", "OLS", "RF")
DISPLAY = {"GB": "GB", "NN": "NN", "OLS": "OLS2", "RF": "RF"}
ESTIMATORS = {"OLS1", "OLS2_GCOMP", "GB_GCOMP", "RF_GCOMP", "NN_GCOMP", "GB_IPW",
              "GB_OLS2_AIPW", "GB_GB_AIPW", "GB_RF_AIPW", "GB_NN_AIPW",
              "GB_OLS2_TMLE", "GB_GB_TMLE", "GB_RF_TMLE", "GB_NN_TMLE", "BCF_GCOMP"}


def render_table(summary):
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Mean relative MSE regret (\%) across the 100 reference datasets in each setting. "
             r"Within each dataset, regret is the percentage excess target-benchmark MSE over the "
             r"smallest target MSE among all 15 estimators; these percentages are then averaged over the 100 reference datasets. So 0 means matching the target-best MSE, and 100 means twice that MSE. "
             r"G-comp uses the associated G-computation estimator directly. All and Excl. use the "
             r"estimator minimizing simulator MSE among all estimators or among non-aligned estimators, "
             r"respectively. "
             r"Bold marks the lowest value within each simulator--setting triplet, including ties. "
             r"WGAN is omitted because it has no associated G-comp in the candidate set.}",
             r"\label{tab:gcomp-selection}", r"{",
             r"\tablebox{%", r"\begin{tabular}{l*{12}{r}}", r"\toprule"]
    lines.append("Generator & " + " & ".join(r"\multicolumn{3}{c}{" + DISPLAY[s] + "}" for s in SIMULATORS) + r" \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
    lines.append(" & " + " & ".join(["G-comp & All & Excl."] * 4) + r" \\")
    lines.append(r"\midrule")
    for index, (design, setting, n, _) in enumerate(SETTINGS):
        if index == 4:
            lines.append(r"\addlinespace[\tablepanelgap]")
        label = "ACIC " + setting if design == "ACIC 2017" else rf"N\&W {setting}, $n={n}$"
        cells = []
        for simulator in SIMULATORS:
            rows = summary.loc[summary.setting_index.eq(index) & summary.generator.eq(simulator)].set_index('selection')
            assert np.isclose(rows.loc['All', 'mean_gcomp_relative_regret_pct'],
                              rows.loc['Excl', 'mean_gcomp_relative_regret_pct'])
            values = [rows.loc['All', 'mean_gcomp_relative_regret_pct'],
                      rows.loc['All', 'mean_selected_relative_regret_pct'],
                      rows.loc['Excl', 'mean_selected_relative_regret_pct']]
            displayed = [f"{value:.1f}" for value in values]
            lowest = min(map(float, displayed))
            cells.extend(rf"\textbf{{{value}}}" if float(value) == lowest else value for value in displayed)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", "}", r"\end{table*}", ""])
    (TABLE_DIR / "results_gcomp_selection.tex").write_text("\n".join(lines), encoding="utf-8")
    from add_setting_win_counts import regret_counts
    regret_counts()


def main():
    summaries, details = [], []
    for index, (design, setting, n, analysis_dir) in enumerate(SETTINGS):
        source = pd.read_csv(analysis_dir / "promotion_detail.csv")
        for simulator in SIMULATORS:
            group = source.loc[source.generator.eq(simulator)]
            assert set(group.seed) == set(range(2001, 2101)), (analysis_dir, simulator)
            assert group.groupby("seed").estimator.apply(set).map(lambda s: s == ESTIMATORS).all()
        summary = write_gcomp_selection_comparison(source, analysis_dir)
        detail = pd.read_csv(analysis_dir / "gcomp_selection_detail.csv")
        assert len(summary) == 8 and len(detail) == 800
        assert summary.n_reference_datasets.eq(100).all()
        assert summary.n_relative_regret.eq(100).all()
        assert np.allclose(summary[["gcomp_win_pct", "selected_win_pct", "tie_pct"]].sum(axis=1), 100)
        assert not detail.loc[detail.selection.eq("Excl"), "selected_same_as_gcomp"].any()
        for frame in (summary, detail):
            frame.insert(0, "setting_index", index)
            frame.insert(1, "design", design)
            frame.insert(2, "setting", setting)
            frame.insert(3, "sample_size", n)
        summaries.append(summary)
        details.append(detail)
    combined = pd.concat(summaries, ignore_index=True)
    full_detail = pd.concat(details, ignore_index=True)
    combined.to_csv(TABLE_DIR / "gcomp_selection_summary.csv", index=False)
    full_detail.to_csv(TABLE_DIR / "gcomp_selection_by_seed.csv", index=False)
    render_table(combined)
    print(combined[["design", "setting", "sample_size", "generator", "selection",
                    "mean_gcomp_relative_regret_pct", "mean_selected_relative_regret_pct"]].to_string(index=False))
    print("\nFamily summaries (equal number of reference datasets in each setting):")
    print(combined.groupby(["design", "generator", "selection"])[
        ["mean_gcomp_relative_regret_pct", "mean_selected_relative_regret_pct"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
