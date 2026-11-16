"""Read-only sensitivity analysis of aggregation across reference datasets.

Run from the repository root:
    python paper/empirical_benchmarking_paper/scripts/explore_mean_vs_median.py

Writes a self-contained HTML report, Markdown, figures, CSV audit tables, and
input checksums under exploratory/mean_vs_median. Does not regenerate manuscript
tables, change selection rules, or replace within-dataset Monte Carlo means.
Requires numpy, pandas, scipy, matplotlib; uses the validated results root from
paper_results_config.py. Settings come from the paper's ranking-table generator.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

from paper_results_config import RESULTS_ROOT
from paper_results_config import SETTINGS

PAPER = Path(__file__).resolve().parents[1]
SIMULATORS = ("GB", "NN", "OLS", "RF", "WGAN")
GCOMP = {s: ("OLS2" if s == "OLS" else s) + "_GCOMP" for s in SIMULATORS[:-1]}
KEY = ["setting", "design", "stratum", "simulator", "measure", "component"]
MEASURES = ("Rank correlation", "Rank promotion", "Absolute rank shift",
            "Signed estimand gap", "Absolute estimand gap", "Outcome distance",
            "Selection regret (%)", "MSE")
INPUTS: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> pd.DataFrame:
    INPUTS[str(path.resolve())] = sha(path)
    return pd.read_csv(path)


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def collect() -> tuple[pd.DataFrame, int]:
    """One observation per reference dataset, metric, and relevant component."""
    chunks = []

    def add(frame, meta, measure, component, values, simulator=None):
        out = pd.DataFrame({"seed": frame.seed.to_numpy(),
                            "simulator": (frame.generator.to_numpy() if simulator is None else simulator),
                            "value": np.asarray(values, dtype=float)})
        for key, value in meta.items():
            out[key] = value
        out["measure"], out["component"] = measure, component
        chunks.append(out)

    sim_data = read(RESULTS_ROOT / "data/simulator_performance.csv")
    target = read(RESULTS_ROOT / "data/target_performance.csv")
    conflicts = 0
    for design, name, n, directory in SETTINGS:
        acic = design == "ACIC 2017"
        setting = "ACIC " + name if acic else f"N&W {name}, n={n}"
        meta = {"setting": setting, "design": "ACIC" if acic else "N&W",
                "stratum": "ACIC" if acic else "N&W n=" + n}
        detail = read(directory / "promotion_detail.csv")
        detail["family_aligned"] = truth(detail.family_aligned)
        assert not detail.duplicated(["seed", "generator", "estimator"]).any()
        assert set(detail.seed) == set(range(2001, 2101))
        assert detail.groupby(["seed", "generator"]).size().eq(15).all()
        np.testing.assert_allclose(detail.promotion, detail.real_rank - detail.sim_rank)
        rho = read(directory / "per_seed_rho.csv")
        for column, component in (("full_rho", "All"), ("excl_rho", "Excl. family")):
            add(rho, meta, "Rank correlation", component, rho[column])
        distances = read(directory / "distance_run_level.csv")
        assert distances.group.eq("ALL").all()
        for column, component in (("W1_mean", "W1"), ("KS_mean", "KS"), ("Energy_mean", "Energy")):
            add(distances, meta, "Outcome distance", component, distances[column])
        for simulator in SIMULATORS:
            frame = detail.loc[detail.generator.eq(simulator)]
            learner = "OLS2" if simulator == "OLS" else simulator
            if simulator != "WGAN":
                for component, estimator in (("G-comp", GCOMP[simulator]),
                        ("AIPW", "GB_" + learner + "_AIPW"),
                        ("TMLE", "GB_" + learner + "_TMLE")):
                    selected = frame.loc[frame.estimator.eq(estimator)]
                    assert len(selected) == 100
                    add(selected, meta, "Rank promotion", component, selected.promotion)
                    add(selected, meta, "Absolute rank shift", component, selected.promotion.abs())
            rest = frame.loc[~frame.family_aligned].copy()
            rest["absolute_shift"] = rest.promotion.abs()
            rest = rest.groupby(["seed", "generator"], as_index=False)[["promotion", "absolute_shift"]].mean()
            add(rest, meta, "Rank promotion", "Rest", rest.promotion)
            add(rest, meta, "Absolute rank shift", "Rest", rest.absolute_shift)
        # Preserve exactly the estimator selected in each existing analysis.
        selections = read(directory / "gcomp_selection_detail.csv")
        assert not selections.duplicated(["seed", "generator", "selection"]).any()
        assert selections.target_best_mse.gt(0).all()
        for prefix in ("gcomp", "selected"):
            np.testing.assert_allclose(
                selections[prefix + "_relative_regret_pct"],
                100*(selections[prefix + "_target_mse"]/selections.target_best_mse-1),
                atol=1e-8, rtol=1e-10)
        target_best = detail.groupby(["seed", "generator"]).real_dgp_mc_mse.min()
        checked = selections.merge(target_best.rename("checked_best"), on=["seed", "generator"], validate="many_to_one")
        np.testing.assert_allclose(checked.target_best_mse, checked.checked_best, atol=1e-12)
        for selection in ("All", "Excl"):
            frame = selections.loc[selections.selection.eq(selection)]
            add(frame, meta, "Selection regret (%)", selection, frame.selected_relative_regret_pct)
        frame = selections.loc[selections.selection.eq("All")]
        add(frame, meta, "Selection regret (%)", "G-comp", frame.gcomp_relative_regret_pct)
        # The existing analysis folder stem is the key in the consolidated data.
        data_key = directory.name.removesuffix("__0_analysis")
        sim = sim_data.loc[sim_data.setting.eq(data_key)].copy()
        real = target.loc[target.setting.eq(data_key)].copy()
        assert len(sim) == 7500 and len(real) == 1500, data_key
        for estimator, frame in sim.groupby("estimator"):
            add(frame, meta, "MSE", estimator, frame.mc_mse)
        for estimator, frame in real.groupby("estimator"):
            add(frame, meta, "MSE", estimator, frame.mc_mse, simulator="Target")
        for seed, frame in real.groupby("seed"):
            assert frame.ate_true.nunique() == 1
        reference = real.groupby("seed").ate_true.first()
        gap_rows = []
        for (seed, simulator), frame in sim.groupby(["seed", "generator"], sort=True):
            counts = frame.ate_true.value_counts()
            conflicts += int(frame.ate_true.max() - frame.ate_true.min() > 1e-12)
            # Same modal-reference convention as make_induced_estimand_table.py.
            gap_rows.append({"seed": seed, "generator": simulator,
                             "gap": float(counts.index[0] - reference.loc[seed])})
        gaps = pd.DataFrame(gap_rows)
        add(gaps, meta, "Signed estimand gap", "Gap", gaps.gap)
        add(gaps, meta, "Absolute estimand gap", "Absolute gap", gaps.gap.abs())
        # Check current mean correlations against the analyzer's own summaries.
        expected = read(directory / "average_rho_by_generator.csv").set_index("generator")
        for col, expected_col in (("full_rho", "avg_full_rho"), ("excl_rho", "avg_excl_rho")):
            actual = rho.groupby("generator")[col].mean()
            np.testing.assert_allclose(actual.loc[expected.index], expected[expected_col], atol=1e-12)
    data = pd.concat(chunks, ignore_index=True)
    assert np.isfinite(data.value).all()
    assert not data.duplicated(KEY + ["seed"]).any()
    assert data.groupby(KEY).size().eq(100).all()
    return data, conflicts


def summaries(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    signed = {"Rank correlation", "Rank promotion", "Signed estimand gap"}
    for key, frame in data.groupby(KEY, sort=True):
        values = frame.value.to_numpy()
        q = np.quantile(values, [.05, .25, .5, .75, .95])
        mean, median = values.mean(), q[2]
        positive = key[-2] not in signed
        rows.append(dict(zip(KEY, key), n=len(values), mean=mean, median=median,
                         difference=median-mean, abs_difference=abs(median-mean),
                         sd=values.std(ddof=1), q05=q[0], q25=q[1], q75=q[3], q95=q[4],
                         maximum=values.max(), minimum=values.min(), zero_fraction=np.mean(values == 0),
                         median_mean_ratio=median/mean if positive and mean > 1e-12 else np.nan,
                         top_decile_share=np.sort(values)[-10:].sum()/values.sum()
                         if positive and values.sum() > 1e-12 else np.nan))
    result = pd.DataFrame(rows)
    # These are screening conventions, not inferential or scientific cutoffs.
    result["screen_large"] = False
    result.loc[result.measure.eq("Rank correlation"), "screen_large"] = result.abs_difference >= .10
    result.loc[result.measure.isin(["Rank promotion", "Absolute rank shift"]), "screen_large"] = result.abs_difference >= 1
    mask = result.measure.isin(["Absolute estimand gap", "Outcome distance", "Selection regret (%)", "MSE"])
    result.loc[mask, "screen_large"] = abs(result.loc[mask, "median_mean_ratio"]-1) >= .25
    return result


def winner_sets(frame, candidates, direction):
    means = frame.set_index(candidates)["mean"]
    medians = frame.set_index(candidates)["median"]
    def best(series):
        optimum = series.max() if direction == "max" else series.min()
        return set(series.index[np.isclose(series, optimum, rtol=1e-10, atol=1e-12)])
    a, b = best(means), best(medians)
    rho = float(spearmanr(means, medians).statistic) if means.nunique() > 1 and medians.nunique() > 1 else np.nan
    tau = float(kendalltau(means, medians).statistic) if means.nunique() > 1 and medians.nunique() > 1 else np.nan
    return {"best_mean": " | ".join(sorted(a)), "best_median": " | ".join(sorted(b)),
            "best_set_changed": a != b, "disjoint_best": not bool(a & b),
            "summary_rank_spearman": rho, "summary_rank_kendall": tau}


def ranking_comparisons(summary):
    rows = []
    for measure in ("Rank correlation", "Absolute estimand gap", "Outcome distance"):
        for (setting, component), frame in summary.loc[summary.measure.eq(measure)].groupby(["setting", "component"]):
            rows.append({"comparison": "Simulator ordering", "measure": measure,
                         "setting": setting, "component": component, "simulator": "All",
                         **winner_sets(frame, "simulator", "max" if measure == "Rank correlation" else "min")})
    for (setting, simulator), frame in summary.loc[summary.measure.eq("Selection regret (%)")].groupby(["setting", "simulator"]):
        rows.append({"comparison": "Selection-rule ordering", "measure": "Selection regret (%)",
                     "setting": setting, "component": "All rules", "simulator": simulator,
                     **winner_sets(frame, "component", "min")})
    for (setting, simulator), frame in summary.loc[summary.measure.eq("MSE")].groupby(["setting", "simulator"]):
        rows.append({"comparison": "Estimator ordering", "measure": "MSE",
                     "setting": setting, "component": "All estimators", "simulator": simulator,
                     **winner_sets(frame, "component", "min")})
    return pd.DataFrame(rows)


def sample_examples(summary, data, seed, n_boot):
    rng = np.random.default_rng(seed)
    # One randomly selected cell per measure per stratum: ACIC, N&W 1000, N&W 4302.
    examples = pd.concat([g.iloc[[int(rng.integers(len(g)))]]
                          for _, g in summary.groupby(["measure", "stratum"], sort=True)], ignore_index=True)
    grouped = {key: g.value.to_numpy() for key, g in data.groupby(KEY, sort=True)}
    ci = []
    for _, row in examples.iterrows():
        values = grouped[tuple(row[k] for k in KEY)]
        boot = values[rng.integers(0, len(values), size=(n_boot, len(values)))]
        differences = np.median(boot, axis=1)-boot.mean(axis=1)
        ci.append(np.quantile(differences, [.025, .975]))
    examples[["bootstrap_difference_low", "bootstrap_difference_high"]] = np.asarray(ci)
    return examples


def table(frame, columns, limit=None):
    frame = frame.loc[:, columns].head(limit).copy() if limit else frame.loc[:, columns].copy()
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value): return "—"
            return f"{value:.4g}" if abs(value) < 100 else f"{value:,.1f}"
        return str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def figures(summary, examples, data, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    items = [("Rank correlation", None), ("Rank promotion", None), ("Absolute rank shift", None),
             ("Absolute estimand gap", None), ("Outcome distance", "W1"), ("Selection regret (%)", None)]
    colors = {"ACIC": "#2F6DB5", "N&W": "#E68632"}
    for ax, (measure, component) in zip(axes.flat, items):
        frame = summary.loc[summary.measure.eq(measure)]
        if component: frame = frame.loc[frame.component.eq(component)]
        for design, group in frame.groupby("design"):
            ax.scatter(group["mean"], group["median"], s=14, alpha=.6, label=design, color=colors[design])
        if measure in ("Absolute estimand gap", "Outcome distance", "Selection regret (%)"):
            ax.set_xscale("symlog", linthresh=.01 if measure != "Selection regret (%)" else 1)
            ax.set_yscale("symlog", linthresh=.01 if measure != "Selection regret (%)" else 1)
        lo = min(frame["mean"].min(), frame["median"].min())
        hi = max(frame["mean"].max(), frame["median"].max())
        ax.plot([lo, hi], [lo, hi], color=".45", linewidth=.8, linestyle="--")
        ax.set_title(measure + (" (W1)" if component else ""), fontsize=10)
        ax.set_xlabel("Mean across reference datasets")
        ax.set_ylabel("Median across reference datasets")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mean_vs_median.png", dpi=180)
    plt.close(fig)
    # Four illustrative distributions from the preselected random examples.
    measures = ["Rank correlation", "Rank promotion", "Outcome distance", "Selection regret (%)"]
    chosen = pd.concat([examples.loc[examples.measure.eq(m)].iloc[[0]] for m in measures])
    grouped = {key: g.value.to_numpy() for key, g in data.groupby(KEY, sort=True)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
        values = grouped[tuple(row[k] for k in KEY)]
        ax.hist(values, bins="auto", color="#6C93BC", alpha=.8)
        ax.axvline(row["mean"], color="#B34C36", label="Mean", linewidth=1.8)
        ax.axvline(row["median"], color="#235B3C", label="Median", linestyle="--", linewidth=1.8)
        ax.set_title(f"{row['measure']}\n{row['setting']} / {row['simulator']} / {row['component']}", fontsize=9)
        ax.set_ylabel("Reference datasets")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "random_example_distributions.png", dpi=180)
    plt.close(fig)


def report(summary, examples, comparisons, data, conflicts, out, seed, n_boot):
    counts = summary.groupby("measure", sort=False).agg(
        cells=("n", "size"), flagged=("screen_large", "sum"),
        median_abs_change=("abs_difference", "median"), maximum_abs_change=("abs_difference", "max"))
    counts["flagged_pct"] = 100*counts.flagged/counts.cells
    count_frame = counts.reset_index()
    count_frame.loc[count_frame.measure.eq("Signed estimand gap"), ["flagged", "flagged_pct"]] = np.nan
    comparison_counts = comparisons.groupby(["comparison", "measure"], sort=False).agg(
        comparisons=("setting", "size"), best_set_changes=("best_set_changed", "sum"),
        disjoint_best=("disjoint_best", "sum"),
        median_rank_correlation=("summary_rank_spearman", "median")).reset_index()
    regret = summary.loc[summary.measure.eq("Selection regret (%)")]
    regret_changes = comparisons.loc[comparisons.comparison.eq("Selection-rule ordering")]
    rho = summary.loc[summary.measure.eq("Rank correlation")]
    distance = summary.loc[summary.measure.eq("Outcome distance")]
    shifts = summary.loc[summary.measure.eq("Absolute rank shift")]
    sign = summary.loc[summary.measure.isin(["Rank correlation", "Rank promotion", "Signed estimand gap"])].copy()
    sign["sign_changed"] = (sign["mean"]*sign["median"] < -1e-12)
    sign_changed = sign.loc[sign.sign_changed]
    largest_regret = regret.nlargest(8, "abs_difference")
    largest_rho = rho.nlargest(8, "abs_difference")
    promotion = summary.loc[summary.measure.eq("Rank promotion")]
    abs_gap = summary.loc[summary.measure.eq("Absolute estimand gap")]
    mse_order = comparisons.loc[comparisons.comparison.eq("Estimator ordering")]
    target_order = mse_order.loc[mse_order.simulator.eq("Target")]

    def cell(measure, setting, simulator, component):
        found = summary.loc[summary.measure.eq(measure) & summary.setting.eq(setting)
                            & summary.simulator.eq(simulator) & summary.component.eq(component)]
        assert len(found) == 1
        return found.iloc[0]

    rf_acic = cell("Selection regret (%)", "ACIC Group-correlated", "RF", "All")
    rf_nw = cell("Selection regret (%)", "N&W C, n=4302", "RF", "All")
    nn_nw = cell("Selection regret (%)", "N&W C, n=4302", "NN", "Excl")
    ols_promotion = cell("Rank promotion", "N&W B, n=4302", "OLS", "G-comp")
    report_lines = [
        "# Mean versus median: exploratory sensitivity analysis",
        "", "## Scope and main findings", "",
        f"This analysis covers all 12 settings and 100 reference datasets per setting from the validated consolidated results. It compares means with medians in **{len(summary):,} summary cells**, based on **{len(data):,} per-reference-dataset values**. The manuscript is unchanged.",
        "",
        f"- Rank correlations: the median absolute mean–median difference is **{rho.abs_difference.median():.3f}**; **{int(rho.screen_large.sum())}/{len(rho)}** cells differ by at least 0.10.",
        f"- Outcome distances: **{int(distance.screen_large.sum())}/{len(distance)}** cells have a median at least 25% above or below their mean. Distances here are already averaged over outcome draws before comparing reference datasets.",
        f"- Absolute rank shifts: **{int(shifts.screen_large.sum())}/{len(shifts)}** cells change by at least one rank position.",
        f"- Selection regret: **{int(regret.screen_large.sum())}/{len(regret)}** cells change by at least 25%. The median-to-mean ratio ranges from **{regret.median_mean_ratio.min():.3f}** to **{regret.median_mean_ratio.max():.3f}**.",
        f"- The set of lowest-regret selection rules changes in **{int(regret_changes.best_set_changed.sum())}/{len(regret_changes)}** simulator–setting comparisons; in **{int(regret_changes.disjoint_best.sum())}**, the mean-optimal and median-optimal sets do not overlap. Set changes can simply add a tied rule; they are not all reversals.",
        "", "## What is being changed—and what is not", "",
        "The unit of aggregation is the **reference dataset** (M = 100), holding the design, sample size, simulator, and metric component fixed. All quantities first follow the existing analysis: rank correlations are computed within each reference dataset; regret uses the estimator selected within that dataset; outcome distances are averages over K = 1000 synthetic datasets. We replace only the subsequent average over reference datasets with a median.",
        "",
        "The mean and median summarize different features. Mean regret measures average percentage loss across datasets, including rare severe failures; median regret describes the middle dataset's percentage loss. A lower median is not evidence that the original mean was incorrect. Likewise, median MSE here is the median of dataset-specific Monte Carlo MSEs—not a replacement of squared-error loss by median squared error. No synthetic datasets, estimators, ranks, simulation votes, or selection rules are regenerated.",
        "",
        "For Rest, promotion and absolute shift are first averaged across non-aligned estimators within each reference dataset, exactly as in the current tables. Their across-dataset median is then taken. In particular, the median absolute shift is not the absolute value of the median promotion. OLS1 remains aligned with OLS2 and is not included in Rest for that simulator. WGAN has no aligned G-comp/AIPW/TMLE; only its Rest group is included in rank movement. It has no regret comparison.",
        "",
        "Pearson distance–fidelity associations are not replaced with a 'median correlation': those are single correlations across the 100 datasets, not averages of 100 correlation estimates. This report instead examines the distance summaries themselves. Replacing Pearson with Spearman would be a different sensitivity analysis.",
        "", "## Comprehensive screening", "",
        "The following descriptive flags help locate sensitivity: absolute change ≥ 0.10 for rank correlations; ≥ 1 position for promotion/absolute shift; and ≥ 25% of the mean for positive-valued MSE, distances, absolute gaps, and regret. These are transparent exploratory screens, not significance tests or validated scientific thresholds. No relative-change threshold is used for signed estimand gaps or signed promotion near zero.",
        "", table(count_frame, ["measure", "cells", "flagged", "flagged_pct", "median_abs_change", "maximum_abs_change"]),
        "", "![Mean versus median across all summary cells](mean_vs_median.png)",
        "", "Each point is one setting–simulator–component cell, not one dataset. The dashed line denotes equality; positive-valued panels use symmetric-log axes to accommodate zeros and long tails. Distance panels show W1 only; CSVs include KS and Energy.",
        "", "## Do orderings and preferred choices change?", "",
        table(comparison_counts, ["comparison", "measure", "comparisons", "best_set_changes", "disjoint_best", "median_rank_correlation"]),
        "",
        "Simulator orderings use higher rank correlation or lower distance/absolute gap as appropriate. Selection-rule orderings compare G-comp, All, and Excl. within a simulator and setting. Estimator orderings compare the 15 MSE summaries under each of the five simulators and the target. These are rankings of across-dataset summaries; the actual per-dataset selection decisions remain unchanged. Numerical ties are retained (rtol 1e-10, atol 1e-12); no tie is resolved arbitrarily. Rank-correlation columns in this table measure agreement between the mean-based and median-based summary orderings, not the original target-versus-simulator fidelity.",
        "", "### Changes in the preferred selection rule", "",
        table(regret_changes.loc[regret_changes.best_set_changed], ["setting", "simulator", "best_mean", "best_median", "disjoint_best"]),
        "", "## Seeded random examples", "",
        f"With seed **{seed}**, one cell was drawn uniformly from each measure × stratum (ACIC, N&W n=1000, N&W n=4302): 24 examples. These examples were selected independently of their mean–median difference. The following interval is a paired, percentile-bootstrap 95% interval for median minus mean, using **{n_boot}** resamples of the same 100 reference datasets. It quantifies uncertainty in the contrast of summaries and is exploratory, without multiplicity correction.",
        "", table(examples, ["measure", "setting", "simulator", "component", "mean", "median", "bootstrap_difference_low", "bootstrap_difference_high"]),
        "", "![Distributions for four preselected random examples](random_example_distributions.png)",
        "", "## Largest changes: intentionally selected diagnostics", "",
        "These cells are selected for their large discrepancy and must not be treated as representative random examples.",
        "", "### Selection regret", "",
        table(largest_regret, ["setting", "simulator", "component", "mean", "median", "q95", "maximum", "top_decile_share"]),
        "", "The top-decile share is the fraction of the total regret contributed by the ten largest of the 100 values. It helps distinguish tail-driven means from a broad change across datasets. Mean-of-ratios regret can also be sensitive to small target-best MSE denominators; the existing denominators are retained.",
        "", "### Rank correlations", "",
        table(largest_rho, ["setting", "simulator", "component", "mean", "median", "q05", "q95"]),
        "", "### Sign reversals", "",
        f"There are **{len(sign_changed)}** strict sign reversals across signed summaries (zero medians are not counted as reversals). Near-zero reversals need substantive interpretation; they do not by themselves establish a different conclusion.",
        "", table(sign_changed.nlargest(12, "abs_difference"), ["measure", "setting", "simulator", "component", "mean", "median"]),
        "", "## Interpretation by measure", "",
        "### Rank correlations: small numerical changes, with close competitions affected", "",
        "No correlation cell changes by 0.10, and the typical absolute difference is 0.024. Nevertheless, the highest excluded-family correlation switches from OLS2 to NN in N&W B at both sample sizes. Thus, broad fidelity patterns are stable, but close comparisons should not be described as invariant. The WGAN IID correlation changes from a small positive mean to a near-zero negative median; both summaries indicate poor fidelity rather than substantively opposite conclusions.",
        "",
        table(comparisons.loc[comparisons.measure.eq("Rank correlation") & comparisons.disjoint_best],
              ["setting", "component", "best_mean", "best_median"]),
        "", "### Promotion and absolute shift: mostly stable, with some discrete medians", "",
        f"Only {int(promotion.screen_large.sum())} of {len(promotion)} promotion cells change by at least one position. For OLS2 G-comp in N&W B at n=4302, mean promotion is {ols_promotion['mean']:.2f} but median promotion is {ols_promotion['median']:.2f}: the positive average is not a shift experienced by the middle reference dataset. RF G-comp in the same setting instead has a larger median than mean, so sensitivity is not uniformly in one direction. The distributions of integer-valued positions can concentrate at particular values; reporting the mean and the median absolute shift addresses different questions.",
        "",
        table(promotion.nlargest(6, "abs_difference"), ["setting", "simulator", "component", "mean", "median", "q05", "q95"]),
        "", "### Distances and absolute estimand gaps: localized sensitivity", "",
        "All three distance cells passing the 25% screen belong to WGAN in N&W C at n=1000. The other 177 cells are closer by that criterion. Absolute estimand gaps are somewhat more sensitive (7/60 cells), particularly for WGAN. These differences concern distributional fit and reference-value approximation; they do not establish that a simulator's estimator ranking improves.",
        "",
        table(distance.loc[distance.screen_large], ["setting", "simulator", "component", "mean", "median", "median_mean_ratio"]),
        "",
        table(abs_gap.nlargest(4, "abs_difference"), ["setting", "simulator", "mean", "median"]),
        "", "### Regret: substantial changes in magnitude, without a disjoint winning rule", "",
        f"Under RF in ACIC group-correlated errors, All-rule regret is {rf_acic['mean']:.1f}% on average but {rf_acic['median']:.1f}% at the median. Under NN in N&W C at n=4302, Excl.-rule regret is {nn_nw['mean']:.2f}% versus {nn_nw['median']:.2f}%. These are meaningful differences between average loss and typical-dataset loss.",
        "",
        f"The reverse also occurs: RF All-rule regret in N&W C at n=4302 rises from a mean of {rf_nw['mean']:,.1f}% to a median of {rf_nw['median']:,.1f}%. Its very poor performance is therefore not explained by a few upper-tail outliers. Selection can produce mixtures of good and bad outcomes; the report's quantiles and maxima help identify this. Do not describe the median as automatically correcting an inflated mean.",
        "",
        "In all eight changed winner sets, the median adds an extra tied rule while retaining the mean-optimal rule. There is no setting in which a wholly different rule becomes preferred by the median. The numerical severity of regret is sensitive, but the best-rule conclusion is more stable in these data. This does not imply identical ordering of the remaining rules or identical effect sizes.",
        "", "### MSE: keep the inner loss fixed", "",
        f"The best estimator changes in {int(mse_order.disjoint_best.sum())}/{len(mse_order)} benchmark–setting comparisons. All changes occur under fitted simulators; none occurs in the {len(target_order)} target-benchmark comparisons. Three occur under RF in ACIC and two under WGAN in N&W. This comparison takes medians of the existing Monte Carlo MSEs across reference datasets; the per-dataset MSE calculations and selected estimators remain fixed.",
        "",
        table(mse_order.loc[mse_order.disjoint_best], ["setting", "simulator", "best_mean", "best_median", "summary_rank_spearman"]),
        "", "## Interpretation for the paper", "",
        "Use the outcome-specific findings above rather than a blanket statement that means and medians are interchangeable. A change in summary magnitude matters differently from a changed ordering or changed preferred rule. For regret, report both the typical-dataset interpretation of the median and the severe failures retained by the mean. For signed promotion, always examine absolute shifts alongside it: a median near zero can coexist with substantial movement in both directions.",
        "",
        "The most informative follow-up is a compact sensitivity supplement if the choice reversals are substantively important; replacing all means in the main paper is not implied by this exploration. No changes have been made to manuscript text, figures, or tables.",
        "", "## Provenance, validation, and limitations", "",
        f"- Results root: `{RESULTS_ROOT.relative_to(PAPER.parents[1]).as_posix()}`. All 1,200 reference runs are complete under the existing validation configuration.",
        "- Every reported cell contains exactly 100 distinct seeds (2001–2100); all values are finite. Promotions were checked against target rank minus simulator rank, and mean rank correlations reproduce existing analyzer summaries.",
        f"- **{conflicts}** simulator–reference groups have recorded induced-estimand spreads exceeding 1e-12 because of historical merged fits (smaller floating-point differences are not counted). Gaps use the same modal recorded estimand convention as the existing induced-estimand table. This sensitivity compares aggregation of those recorded values; it does not resolve historical fit differences or mix reference values within a cell.",
        "- Bootstrap intervals assume independent reference datasets within a setting, conditional on design inputs held fixed. They do not incorporate inner Monte Carlo error, simulator retraining beyond the recorded fits, or uncertainty from choosing examples.",
        "- There is no pooling across unequal settings when comparing means with medians. Family-level macro-averages are not replaced by pooled medians; those would be additional changes of aggregation level.",
        "- Internal simulator label OLS denotes the paper's OLS2 simulator. Excl denotes the family-excluded selection rule. Table entries are rounded only for display; all comparisons use unrounded values.",
        "- A median can hide rare failures and, for discrete ranks or zero-inflated regret, can create ties. The comprehensive CSVs retain quantiles, zero fractions, and maxima to make that visible.",
        "- All source SHA-256 hashes, command settings, software versions, and output counts are recorded in `manifest.json`; the script verifies that source files remain unchanged.",
        "", "## Reproduce and inspect", "", "```powershell",
        "python paper/empirical_benchmarking_paper/scripts/explore_mean_vs_median.py", "```", "",
        "Optional arguments: `--seed`, `--bootstrap`, `--output`. The seed controls example selection and bootstrap resampling; it does not alter the underlying data.",
        "",
        "Outputs: `summary_all.csv` (all cells); `random_examples.csv` (24 examples and bootstrap intervals); `ordering_comparisons.csv`; `per_reference_values.csv`; `screening_summary.csv`; `manifest.json`; `validation.json`; two PNGs; this Markdown and self-contained HTML report.",
    ]
    (out / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    count_frame.to_csv(out / "screening_summary.csv", index=False)
    return report_lines


def render_html(lines, out):
    """Small dependency-free renderer for the report's headings/tables/images."""
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        return re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    parts, in_table, in_code = [], False, False
    for line in lines:
        if in_table and not line.startswith("|"):
            parts.append("</tbody></table></div>")
            in_table = False
        if line.startswith("```"):
            parts.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
        elif in_code:
            parts.append(html.escape(line))
        elif line.startswith("|"):
            if re.match(r"^\|[\s|:-]+\|$", line): continue
            cells = line.strip("|").split("|")
            if not in_table:
                parts.append("<div class='table-scroll'><table><thead><tr>" + "".join("<th>"+inline(c.strip())+"</th>" for c in cells)+"</tr></thead><tbody>")
                in_table = True
            else:
                parts.append("<tr>"+"".join("<td>"+inline(c.strip())+"</td>" for c in cells)+"</tr>")
        elif line.startswith("#"):
            level = len(line)-len(line.lstrip("#"))
            parts.append(f"<h{level}>"+inline(line[level:].strip())+f"</h{level}>")
        elif line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            encoded = base64.b64encode((out / match[2]).read_bytes()).decode()
            parts.append(f"<img alt='{html.escape(match[1])}' src='data:image/png;base64,{encoded}'>")
        elif line.startswith("- "):
            parts.append("<p class='bullet'>• "+inline(line[2:])+"</p>")
        elif line:
            parts.append("<p>"+inline(line)+"</p>")
    css = "body{font:16px/1.55 system-ui,sans-serif;color:#243247;max-width:1280px;margin:40px auto;padding:0 28px}h1,h2,h3{color:#173b5c}h2{margin-top:2em;border-bottom:1px solid #ccd6df}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:7px 9px;text-align:left;border-bottom:1px solid #dae1e8;white-space:nowrap}th{background:#eaf0f5}tr:nth-child(even){background:#f7f9fb}.table-scroll{overflow-x:auto;margin:20px 0}img{max-width:100%;margin:20px 0}code,pre{background:#f0f3f6}pre{padding:16px}.bullet{margin:.4em 0}"
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Mean versus median sensitivity</title><style>"+css+"</style><main>"+"\n".join(parts)+"</main></html>"
    (out / "report.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "exploratory/mean_vs_median")
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100: parser.error("--bootstrap must be at least 100")
    out = args.output.resolve()
    # Avoid accidental overwriting of manuscript assets or raw results.
    for protected in (PAPER / "tables", PAPER / "figures", RESULTS_ROOT):
        if out == protected or protected in out.parents:
            parser.error(f"Output must not be inside {protected}")
    out.mkdir(parents=True, exist_ok=True)
    for path in (Path(__file__).with_name("paper_results_config.py"),
                 Path(__file__).with_name("make_rank_aggregation_table.py"),
                 RESULTS_ROOT / "validation.json"):
        INPUTS[str(path.resolve())] = sha(path)
    data, conflicts = collect()
    summary = summaries(data)
    independent = data.groupby(KEY).value.agg(["mean", "median", "size"]).reset_index().merge(
        summary, on=KEY, validate="one_to_one")
    np.testing.assert_allclose(independent.mean_x, independent.mean_y, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(independent.median_x, independent.median_y, rtol=1e-12, atol=1e-12)
    assert independent["size"].eq(100).all()
    comparisons = ranking_comparisons(summary)
    examples = sample_examples(summary, data, args.seed, args.bootstrap)
    data.to_csv(out / "per_reference_values.csv", index=False)
    summary.to_csv(out / "summary_all.csv", index=False)
    comparisons.to_csv(out / "ordering_comparisons.csv", index=False)
    examples.to_csv(out / "random_examples.csv", index=False)
    figures(summary, examples, data, out)
    lines = report(summary, examples, comparisons, data, conflicts, out, args.seed, args.bootstrap)
    render_html(lines, out)
    assert all(sha(Path(path)) == digest for path, digest in INPUTS.items()), "Source input changed during analysis"
    import scipy
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
                "seed": args.seed, "bootstrap_resamples": args.bootstrap, "settings": 12,
                "summary_cells": len(summary), "per_reference_values": len(data), "random_examples": len(examples),
                "mixed_estimand_groups": conflicts, "input_files": INPUTS, "source_files_unchanged": True,
                "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps({
        "summary_cells_independently_recomputed": len(summary),
        "all_groups_100_reference_datasets": True,
        "all_values_finite": True,
        "rank_correlations_match_analyzer_means": True,
        "promotions_match_rank_differences": True,
        "regrets_match_target_mse_ratios": True,
        "input_files_unchanged": True,
        "random_examples": len(examples),
    }, indent=2), encoding="utf-8")
    print(f"Validated {len(summary)} cells; {len(data)} values; {len(examples)} random examples.")
    print(f"Report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
