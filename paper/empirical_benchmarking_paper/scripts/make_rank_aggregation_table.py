"""Render per-dataset rank-aggregation results for the empirical paper."""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


from paper_results_config import SETTINGS
TABLE_PATH = Path(__file__).resolve().parents[1] / "tables" / "results_rank_aggregation.tex"
SIMULATORS = ("GB", "NN", "OLS", "RF", "WGAN")
DISPLAY_SIMULATORS = ("GB", "NN", "OLS2", "RF", "WGAN")


def consensus_rhos(analysis_dir: Path, *, exclude_aligned_votes: bool) -> pd.Series:
    detail = pd.read_csv(analysis_dir / "promotion_detail.csv")
    required = {"seed", "generator", "estimator", "real_rank", "sim_rank", "family_aligned"}
    if missing := required.difference(detail.columns):
        raise ValueError(f"{analysis_dir} is missing columns: {sorted(missing)}")

    values: list[float] = []
    seeds = []
    for seed, per_seed in detail.groupby("seed", sort=True):
        available = set(per_seed["generator"].astype(str).unique())
        if missing := set(SIMULATORS).difference(available):
            raise ValueError(f"seed {seed} in {analysis_dir} lacks simulators: {sorted(missing)}")
        votes = per_seed.loc[~per_seed["family_aligned"].astype(bool)].copy() if exclude_aligned_votes else per_seed
        mean_rank = votes.groupby("estimator", sort=False)["sim_rank"].mean()
        real_ranks = per_seed.drop_duplicates("estimator").set_index("estimator").loc[mean_rank.index, "real_rank"]
        if mean_rank.isna().any() or real_ranks.isna().any():
            raise ValueError(f"seed {seed} in {analysis_dir} has incomplete ranks")
        values.append(float(spearmanr(real_ranks, mean_rank.rank(method="average", ascending=True)).statistic))
        seeds.append(seed)
    return pd.Series(values, index=seeds, dtype=float)


def individual_mean_rhos(analysis_dir: Path) -> dict[str, float]:
    summary = pd.read_csv(analysis_dir / "average_rho_by_generator.csv").set_index("generator")
    values: dict[str, float] = {}
    for simulator in SIMULATORS:
        values[f"{simulator}_all"] = float(summary.loc[simulator, "avg_full_rho"])
        values[f"{simulator}_excl"] = float(summary.loc[simulator, "avg_excl_rho"])
    return values


HEADER = "\\begin{table*}[!t]\n\\centering\n\\caption{Mean Spearman rank correlations and absolute estimand differences across 100 reference datasets per setting. All and Excl. use full and alignment-excluded rankings; aggregation follows Section~\\ref{sec:benchmarking-procedure}. For each generator, bold, underlining, and gray shading mark the highest All correlation, highest Excl. correlation, and smallest mean absolute estimand difference, respectively.}\n\\label{tab:rank-aggregation}\n{\n\n\n\\makebox[\\textwidth][c]{%\n\\tablebox{%\n\\begin{tabular}{l@{\\hspace{\\tablegroupsep}}ccc@{\\hspace{\\tablegroupsep}}ccc@{\\hspace{\\tablegroupsep}}ccc@{\\hspace{\\tablegroupsep}}ccc@{\\hspace{\\tablegroupsep}}cc@{\\hspace{\\tablegroupsep}}cc}\n\\toprule\n& \\multicolumn{3}{c}{GB} & \\multicolumn{3}{c}{NN} & \\multicolumn{3}{c}{OLS2} & \\multicolumn{3}{c}{RF} & \\multicolumn{2}{c}{WGAN} & \\multicolumn{2}{c}{Aggregated} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\\cmidrule(lr){14-15}\\cmidrule(lr){16-17}\nGenerator & All & Excl. & \\(\\overline{|\\Delta\\Phi_s|}\\) & All & Excl. & \\(\\overline{|\\Delta\\Phi_s|}\\) & All & Excl. & \\(\\overline{|\\Delta\\Phi_s|}\\) & All & Excl. & \\(\\overline{|\\Delta\\Phi_s|}\\) & All & \\(\\overline{|\\Delta\\Phi_s|}\\) & All & Excl. \\\\\n"


def main() -> None:
    estimands = pd.read_csv(TABLE_PATH.parent / "induced_estimands_by_seed.csv")
    rows = []
    per_seed_rows = []
    for index, (design, setting, sample_size, analysis_dir) in enumerate(SETTINGS):
        all_votes = consensus_rhos(analysis_dir, exclude_aligned_votes=False)
        excluded_votes = consensus_rhos(analysis_dir, exclude_aligned_votes=True)
        source_setting = estimands["setting"].drop_duplicates().iloc[index]
        source = estimands.loc[estimands.setting.eq(source_setting)].set_index("seed")
        assert len(source) == 100
        for seed in all_votes.index:
            per_seed_rows.append({'setting': source_setting, 'seed': seed,
                'aggregate_all': all_votes.loc[seed],
                'aggregate_excl': excluded_votes.loc[seed],
                'wgan_induced_estimand': source.loc[seed, 'WGAN'],
                'reference_estimand': source.loc[seed, 'Reference'],
                'wgan_estimand_gap': source.loc[seed, 'WGAN_gap']})
        row = {"design": design, "setting": setting, "sample_size": sample_size,
               "aggregate_all": all_votes.mean(), "aggregate_excl": excluded_votes.mean(),
               **individual_mean_rhos(analysis_dir)}
        for simulator, display in zip(SIMULATORS, DISPLAY_SIMULATORS):
            row[simulator + "_gap"] = source[display + "_gap"].mean()
            row[simulator + "_abs_gap"] = source[display + "_gap"].abs().mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_PATH.parent / "rank_aggregation_summary.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(TABLE_PATH.parent / "rank_aggregation_by_seed.csv", index=False)
    render_table(summary)


def render_table(summary: pd.DataFrame) -> None:
    lines = HEADER.splitlines() + [r"\midrule"]
    for row in summary.to_dict("records"):
        all_values = [row[s + "_all"] for s in SIMULATORS] + [row["aggregate_all"]]
        excl_values = [row[s + "_excl"] for s in SIMULATORS] + [row["aggregate_excl"]]
        eligible_excl_values = [row[s + "_excl"] for s in SIMULATORS if s != "WGAN"] + [row["aggregate_excl"]]
        smallest_gap = min(row[s + "_abs_gap"] for s in SIMULATORS)
        cells = []
        for i, simulator in enumerate(SIMULATORS):
            a, e = all_values[i], excl_values[i]
            cells.append(rf"\textbf{{{a:.3f}}}" if a == max(all_values) else f"{a:.3f}")
            if simulator != "WGAN":
                cells.append(rf"\underline{{{e:.3f}}}" if e == max(eligible_excl_values) else f"{e:.3f}")
            gap = row[simulator + '_abs_gap']
            shade = r"\cellcolor{gray!20}" if gap == smallest_gap else ""
            cells.append(shade + f"{gap:.3f}")
        a, e = all_values[-1], excl_values[-1]
        cells.extend([rf"\textbf{{{a:.3f}}}" if a == max(all_values) else f"{a:.3f}",
                      rf"\underline{{{e:.3f}}}" if e == max(eligible_excl_values) else f"{e:.3f}"])
        design = "ACIC" if row["design"] == "ACIC 2017" else r"N\&W"
        setting = row["setting"] if row["sample_size"] == "--" else f"{row['setting']}, $n={row['sample_size']}$"
        lines.append(design + " " + setting + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", "}", "}", r"\end{table*}", ""])
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    from add_setting_win_counts import rank_counts
    rank_counts()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
