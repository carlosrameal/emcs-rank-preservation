from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


FAMILY_MAP: dict[str, set[str]] = {
    "GB": {"GB_GCOMP", "GB_GB_AIPW", "GB_GB_TMLE"},
    "RF": {"RF_GCOMP", "GB_RF_AIPW", "GB_RF_TMLE"},
    "NN": {"NN_GCOMP", "GB_NN_AIPW", "GB_NN_TMLE"},
    "OLS": {"OLS1", "OLS2_GCOMP", "GB_OLS2_AIPW", "GB_OLS2_TMLE"},
    "WGAN": set(),
}

GCOMP_BY_SIMULATOR = {
    "OLS": "OLS2_GCOMP", "GB": "GB_GCOMP", "RF": "RF_GCOMP", "NN": "NN_GCOMP",
}


def gcomp_selection_comparison(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare associated G-comp with simulator selection, using target MSEs.

    Excl removes aligned estimators from the selection pool only. The associated
    G-comp remains the baseline. Exact simulator-MSE ties are broken by estimator
    name, as in the ranking analysis, without consulting target performance.
    Target-MSE ties use only numerical tolerance (rtol=1e-12, atol=1e-15).
    Regret subtracts the minimum target MSE across ALL available estimators for
    that reference dataset, including aligned estimators even in Excl. Relative
    regret is computed within dataset, then averaged (not a ratio of averages).
    Zero target minima give undefined relative regret, never an invented epsilon.
    WGAN and incomplete extensions without the baseline are not eligible.
    """
    columns = ["seed", "generator", "selection", "gcomp_estimator", "selected_estimator",
               "n_candidates", "n_simulator_minima", "selected_same_as_gcomp",
               "gcomp_target_mse", "selected_target_mse", "target_mse_difference", "outcome",
               "target_best_estimator", "target_best_mse", "gcomp_regret", "selected_regret",
               "gcomp_relative_regret_pct", "selected_relative_regret_pct",
               "relative_regret_difference_pp"]
    summary_columns = ["generator", "selection", "n_reference_datasets", "gcomp_wins",
                       "selected_wins", "ties", "gcomp_win_pct", "selected_win_pct", "tie_pct",
                       "same_estimator_pct", "mean_target_mse_difference", "n_relative_regret",
                       "mean_target_best_mse", "mean_gcomp_regret", "mean_selected_regret",
                       "mean_gcomp_relative_regret_pct", "mean_selected_relative_regret_pct",
                       "mean_relative_regret_difference_pp"]
    rows = []
    if not detail.empty:
        required = {"seed", "generator", "estimator", "simulator_mc_mse", "real_dgp_mc_mse"}
        if missing := required - set(detail.columns):
            raise ValueError(f"Selection comparison missing columns: {sorted(missing)}")
        if detail.duplicated(["seed", "generator", "estimator"]).any():
            raise ValueError("Duplicate seed/simulator/estimator in selection comparison")
        target_rows = detail[["seed", "estimator", "real_dgp_mc_mse"]].drop_duplicates()
        if target_rows.duplicated(["seed", "estimator"]).any():
            raise ValueError("Inconsistent target MSE across simulators for the same seed and estimator")
        target_values = target_rows.real_dgp_mc_mse.to_numpy(dtype=float)
        if not np.isfinite(target_values).all() or (target_values < 0).any():
            raise ValueError("Invalid target MSE in regret comparison")
        best = (target_rows.sort_values(["real_dgp_mc_mse", "estimator"], kind="stable")
                .drop_duplicates("seed").set_index("seed"))
        for (seed, simulator), group in detail.groupby(["seed", "generator"], sort=True):
            baseline = GCOMP_BY_SIMULATOR.get(str(simulator))
            if baseline is None or baseline not in set(group.estimator):
                continue
            values = group[["simulator_mc_mse", "real_dgp_mc_mse"]].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid MSE for selection comparison: {seed}, {simulator}")
            baseline_mse = float(group.loc[group.estimator.eq(baseline), "real_dgp_mc_mse"].iloc[0])
            best_mse = float(best.loc[seed, "real_dgp_mc_mse"])
            best_estimator = str(best.loc[seed, "estimator"])
            for selection in ("All", "Excl"):
                candidates = group if selection == "All" else group.loc[
                    ~group.estimator.map(lambda e: _is_family_aligned(simulator, e))]
                if candidates.empty:
                    continue
                ordered = candidates.sort_values(["simulator_mc_mse", "estimator"], kind="stable")
                chosen = ordered.iloc[0]
                selected_mse = float(chosen.real_dgp_mc_mse)
                difference = selected_mse - baseline_mse
                gcomp_regret = baseline_mse - best_mse
                selected_regret = selected_mse - best_mse
                gcomp_relative = 100 * gcomp_regret / best_mse if best_mse > 0 else np.nan
                selected_relative = 100 * selected_regret / best_mse if best_mse > 0 else np.nan
                tied = np.isclose(selected_mse, baseline_mse, rtol=1e-12, atol=1e-15)
                rows.append({"seed": seed, "generator": simulator, "selection": selection,
                             "gcomp_estimator": baseline, "selected_estimator": chosen.estimator,
                             "n_candidates": len(candidates),
                             "n_simulator_minima": int(candidates.simulator_mc_mse.eq(chosen.simulator_mc_mse).sum()),
                             "selected_same_as_gcomp": chosen.estimator == baseline,
                             "gcomp_target_mse": baseline_mse, "selected_target_mse": selected_mse,
                             "target_mse_difference": difference,
                             "outcome": "tie" if tied else ("gcomp" if difference > 0 else "selected"),
                             "target_best_estimator": best_estimator, "target_best_mse": best_mse,
                             "gcomp_regret": gcomp_regret, "selected_regret": selected_regret,
                             "gcomp_relative_regret_pct": gcomp_relative,
                             "selected_relative_regret_pct": selected_relative,
                             "relative_regret_difference_pp": selected_relative - gcomp_relative})
    comparisons = pd.DataFrame(rows, columns=columns)
    summaries = []
    for (simulator, selection), group in comparisons.groupby(["generator", "selection"], sort=True):
        n = len(group)
        counts = group.outcome.value_counts()
        summaries.append({"generator": simulator, "selection": selection, "n_reference_datasets": n,
                          "gcomp_wins": int(counts.get("gcomp", 0)),
                          "selected_wins": int(counts.get("selected", 0)), "ties": int(counts.get("tie", 0)),
                          "gcomp_win_pct": 100 * counts.get("gcomp", 0) / n,
                          "selected_win_pct": 100 * counts.get("selected", 0) / n,
                          "tie_pct": 100 * counts.get("tie", 0) / n,
                          "same_estimator_pct": 100 * group.selected_same_as_gcomp.mean(),
                          "mean_target_mse_difference": group.target_mse_difference.mean(),
                          "n_relative_regret": int(group.target_best_mse.gt(0).sum()),
                          "mean_target_best_mse": group.target_best_mse.mean(),
                          "mean_gcomp_regret": group.gcomp_regret.mean(),
                          "mean_selected_regret": group.selected_regret.mean(),
                          "mean_gcomp_relative_regret_pct": group.gcomp_relative_regret_pct.mean(),
                          "mean_selected_relative_regret_pct": group.selected_relative_regret_pct.mean(),
                          "mean_relative_regret_difference_pp": group.relative_regret_difference_pp.mean()})
    return comparisons, pd.DataFrame(summaries, columns=summary_columns)


def write_gcomp_selection_comparison(detail: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    comparisons, summary = gcomp_selection_comparison(detail)
    comparisons.to_csv(out_dir / "gcomp_selection_detail.csv", index=False)
    summary.to_csv(out_dir / "gcomp_selection_summary.csv", index=False)
    return summary


def gcomp_only_alignment_metrics(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute alignment diagnostics marking only associated G-comp aligned.

    Correlations are recomputed after removing that estimator from both rank
    vectors. Promotion and absolute shift retain positions in the FULL rankings,
    and are summarized separately for G-comp and the remaining estimators.
    """
    metrics = ["rho_all", "rho_excluded", "promotion_aligned", "promotion_excluded",
               "shift_aligned", "shift_excluded"]
    rows = []
    if not detail.empty:
        if detail.duplicated(["seed", "generator", "estimator"]).any():
            raise ValueError("Duplicate estimator in G-comp-only alignment diagnostics")
        for (seed, simulator), group in detail.groupby(["seed", "generator"], sort=True):
            baseline = GCOMP_BY_SIMULATOR.get(str(simulator))
            aligned = group.estimator.eq(baseline) if baseline is not None else pd.Series(False, index=group.index)
            target = group.real_rank.to_numpy(dtype=float)
            simulated = group.sim_rank.to_numpy(dtype=float)
            if not np.isfinite(target).all() or not np.isfinite(simulated).all():
                raise ValueError(f"Invalid rank in G-comp-only diagnostics: {seed}, {simulator}")
            promotion = group.real_rank - group.sim_rank
            shift = promotion.abs()
            remaining = group.loc[~aligned]
            rows.append({"seed": seed, "generator": simulator,
                         "rho_all": _safe_corr(target, simulated, "spearman"),
                         "rho_excluded": _safe_corr(remaining.real_rank, remaining.sim_rank, "spearman"),
                         "promotion_aligned": promotion.loc[aligned].mean(),
                         "promotion_excluded": promotion.loc[~aligned].mean(),
                         "shift_aligned": shift.loc[aligned].mean(),
                         "shift_excluded": shift.loc[~aligned].mean()})
    per_seed = pd.DataFrame(rows, columns=["seed", "generator", *metrics])
    summary = (pd.DataFrame(columns=["generator", *metrics]) if per_seed.empty else
               per_seed.groupby("generator", as_index=False)[metrics].mean())
    return per_seed, summary


def _is_family_aligned(generator: str, estimator: str) -> bool:
    generator_key = str(generator).upper()
    estimator_key = str(estimator).upper()
    if generator_key in FAMILY_MAP:
        if estimator_key in FAMILY_MAP[generator_key]:
            return True
        if generator_key == "NN":
            return estimator_key.startswith("NN") or estimator_key.startswith("GB_NN")
        if generator_key == "GB":
            # Family alignment is determined by the outcome model, not by the
            # propensity-score model.  In particular, GB_IPW is not GB-family
            # aligned, and GB_RF/GB_NN estimators belong to the RF/NN families.
            return estimator_key in {"GB_GCOMP", "GB_GB_AIPW", "GB_GB_TMLE"}
        if generator_key == "RF":
            return estimator_key.startswith("RF") or estimator_key.startswith("GB_RF")
        if generator_key == "OLS":
            return estimator_key in FAMILY_MAP["OLS"]
    return False


def _parse_args():
    p = argparse.ArgumentParser(
        description="Analyze one-configuration simulator result bundles across seeds using the embedded per-run REAL_DGP benchmark."
    )
    p.add_argument(
        "bundle",
        type=str,
        help="Path to a zip bundle containing many seed zips, or a directory containing seed runs/zips.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for analysis CSVs/TXT. Defaults to <bundle_stem>__0_analysis next to the bundle.",
    )
    p.add_argument(
        "--include-wgan-in-distance-corr",
        action="store_true",
        help="Also compute a second distance-correlation table including WGAN (the no-WGAN table is always produced).",
    )
    p.add_argument('--allow-mixed-estimands', action='store_true',
                   help='Preserve mixed-reference legacy metrics; omit ambiguous simulator estimand gaps.')
    return p.parse_args()


def _default_output_dir(bundle_path: Path) -> Path:
    stem = bundle_path.stem if bundle_path.suffix.lower() == ".zip" else bundle_path.name
    return bundle_path.parent / f"{stem}__0_analysis"


def _seed_from_name(name: str) -> int | None:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else None


def _iter_run_payloads(bundle_path: Path):
    if bundle_path.is_dir():
        direct_summary = bundle_path / "MASTER_mc_summary.csv"
        direct_dist = bundle_path / "MASTER_distribution_distances_summary.csv"
        if direct_summary.exists() and direct_dist.exists():
            yield bundle_path.name, bundle_path
            return

        # zipped runs inside directory
        zipped = sorted(p for p in bundle_path.iterdir() if p.suffix.lower() == ".zip")
        if zipped:
            for p in zipped:
                yield p.name, p.read_bytes()
            return

        # unzipped run folders inside directory
        child_dirs = sorted(x for x in bundle_path.iterdir() if x.is_dir())
        run_dirs = [p for p in child_dirs if (p / "MASTER_mc_summary.csv").exists() and (p / "MASTER_distribution_distances_summary.csv").exists()]
        if run_dirs:
            for p in run_dirs:
                yield p.name, p
            return

        # recursively search for run folders in nested directories
        for p in sorted(x for x in bundle_path.rglob("*") if x.is_dir()):
            if p == bundle_path:
                continue
            if (p / "MASTER_mc_summary.csv").exists() and (p / "MASTER_distribution_distances_summary.csv").exists():
                yield p.name, p
        return

    if bundle_path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported bundle path: {bundle_path}")

    with zipfile.ZipFile(bundle_path) as outer:
        inner_zip_names = sorted(n for n in outer.namelist() if n.lower().endswith(".zip"))
        if inner_zip_names:
            for inner_name in inner_zip_names:
                yield inner_name, outer.read(inner_name)
            return

        run_prefixes = sorted(
            {
                name.split("/", 1)[0]
                for name in outer.namelist()
                if "/" in name and name.split("/", 1)[0] not in {"", "."}
            }
        )
        if run_prefixes:
            for prefix in run_prefixes:
                yield prefix, {"zip_path": bundle_path, "prefix": f"{prefix}/"}
            return

    raise ValueError(f"No inner seed runs found in {bundle_path}")


def _read_member_csv_from_zip(zf: zipfile.ZipFile, candidate_names: list[str]) -> pd.DataFrame:
    names = set(zf.namelist())
    for name in candidate_names:
        if name in names:
            return pd.read_csv(io.BytesIO(zf.read(name)))
    raise KeyError(f"None of the candidate csv paths exist: {candidate_names}")


def _read_embedded_benchmark_from_dir(run_dir: Path) -> pd.DataFrame | None:
    candidates = sorted(run_dir.glob("_dgp_benchmark/*/MASTER_mc_summary.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def _read_embedded_benchmark_from_zip(
    zf: zipfile.ZipFile,
    *,
    prefix: str = "",
) -> pd.DataFrame | None:
    matches = sorted(
        name
        for name in zf.namelist()
        if name.startswith(prefix + "_dgp_benchmark/")
        and name.endswith("/MASTER_mc_summary.csv")
    )
    if not matches:
        return None
    return pd.read_csv(io.BytesIO(zf.read(matches[0])))


def _read_prefixed_csv_from_zip(
    zf: zipfile.ZipFile,
    *,
    prefix: str,
    candidate_names: list[str],
) -> pd.DataFrame:
    prefixed = [prefix + name for name in candidate_names]
    return _read_member_csv_from_zip(zf, prefixed)


def _read_run_csvs(payload) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    summary_candidates = [
        "MASTER_mc_summary.csv",
        "produced_output/untracked/custom_selection/MASTER_mc_summary.csv",
    ]
    dist_candidates = [
        "MASTER_distribution_distances_summary.csv",
        "produced_output/untracked/custom_selection/MASTER_distribution_distances_summary.csv",
    ]

    if isinstance(payload, Path):
        for csv_name in summary_candidates:
            sp = payload / csv_name
            if sp.exists():
                perf = pd.read_csv(sp)
                break
        else:
            raise FileNotFoundError(f"No MASTER_mc_summary.csv found in {payload}")

        for csv_name in dist_candidates:
            dp = payload / csv_name
            if dp.exists():
                dist = pd.read_csv(dp)
                break
        else:
            raise FileNotFoundError(f"No MASTER_distribution_distances_summary.csv found in {payload}")

        benchmark = _read_embedded_benchmark_from_dir(payload)
        return perf, dist, benchmark

    if isinstance(payload, dict):
        zip_path = Path(payload["zip_path"])
        prefix = str(payload["prefix"])
        with zipfile.ZipFile(zip_path) as zf:
            perf = _read_prefixed_csv_from_zip(zf, prefix=prefix, candidate_names=summary_candidates)
            dist = _read_prefixed_csv_from_zip(zf, prefix=prefix, candidate_names=dist_candidates)
            benchmark = _read_embedded_benchmark_from_zip(zf, prefix=prefix)
        return perf, dist, benchmark

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        perf = _read_member_csv_from_zip(zf, summary_candidates)
        dist = _read_member_csv_from_zip(zf, dist_candidates)
        benchmark = _read_embedded_benchmark_from_zip(zf)
    return perf, dist, benchmark


def _build_real_dgp_rank_table(perf_df: pd.DataFrame) -> pd.DataFrame:
    need = {"generator", "estimator", "mc_mse"}
    if not need.issubset(perf_df.columns):
        raise ValueError(f"Benchmark MASTER_mc_summary.csv missing columns {need - set(perf_df.columns)}")
    real_df = perf_df.loc[perf_df["generator"].astype(str) == "REAL_DGP", ["estimator", "mc_mse"]].copy()
    if real_df.empty:
        raise ValueError("Benchmark MASTER_mc_summary.csv has no REAL_DGP rows.")
    real_df = real_df.sort_values(["mc_mse", "estimator"], kind="stable").reset_index(drop=True)
    real_df["rank_mc_mse"] = np.arange(1, len(real_df) + 1)
    return real_df[["rank_mc_mse", "estimator", "mc_mse"]]


def _rank_one_generator(perf_df: pd.DataFrame, generator: str) -> pd.DataFrame:
    g = perf_df.loc[perf_df["generator"].astype(str) == generator, ["estimator", "mc_mse"]].copy()
    g = g.sort_values(["mc_mse", "estimator"], kind="stable").reset_index(drop=True)
    g["sim_rank"] = np.arange(1, len(g) + 1)
    return g


def _safe_corr(x, y, method: str) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(method)


def _unique_estimand(performance_df: pd.DataFrame, generator: str, *, allow_mixed: bool = False) -> float:
    """Return the benchmark-compatible estimand recorded for one generator.

    Every estimator in a Monte Carlo summary is evaluated against the same
    value of ``ate_true``.  Checking that invariant here avoids silently
    attaching an estimator-specific quantity to a simulator-level ranking.
    """
    if "ate_true" not in performance_df.columns:
        return np.nan
    values = performance_df.loc[
        performance_df["generator"].astype(str) == generator, "ate_true"
    ].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan
    if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12):
        if allow_mixed:
            return np.nan
        raise ValueError(
            f"Generator {generator!r} has inconsistent ate_true values within one run."
        )
    return float(values[0])


def _estimand_gap_correlation_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Relate absolute compatible-estimand gaps to per-dataset rank fidelity."""
    columns = [
        "generator", "n_pairs", "mean_abs_estimand_gap", "sd_abs_estimand_gap",
        "pearson_abs_gap_full_rho", "spearman_abs_gap_full_rho",
        "pearson_abs_gap_excl_rho", "spearman_abs_gap_excl_rho",
    ]
    if per_seed_df.empty or "abs_estimand_gap" not in per_seed_df.columns:
        return pd.DataFrame(columns=columns)
    rows = []
    for generator, group in per_seed_df.groupby("generator", sort=True):
        gap = group["abs_estimand_gap"]
        rows.append({
            "generator": generator,
            "n_pairs": int(np.sum(np.isfinite(gap))),
            "mean_abs_estimand_gap": float(np.nanmean(gap)),
            "sd_abs_estimand_gap": float(np.nanstd(gap, ddof=1)),
            "pearson_abs_gap_full_rho": _safe_corr(gap, group["full_rho"], "pearson"),
            "spearman_abs_gap_full_rho": _safe_corr(gap, group["full_rho"], "spearman"),
            "pearson_abs_gap_excl_rho": _safe_corr(gap, group["excl_rho"], "pearson"),
            "spearman_abs_gap_excl_rho": _safe_corr(gap, group["excl_rho"], "spearman"),
        })
    return pd.DataFrame(rows, columns=columns)


def _distance_rho_correlation_table(run_level_df: pd.DataFrame, *, rho_col: str) -> pd.DataFrame:
    metrics = ["W1_mean", "KS_mean", "Energy_mean", "MeanDiff_mean", "StdRatio_mean"]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "distance_metric": metric,
                "pearson_with_rho": _safe_corr(run_level_df[metric], run_level_df[rho_col], "pearson"),
                "spearman_with_rho": _safe_corr(run_level_df[metric], run_level_df[rho_col], "spearman"),
                "n_pairs": int(np.sum(np.isfinite(run_level_df[metric]) & np.isfinite(run_level_df[rho_col]))),
            }
        )
    return pd.DataFrame(rows)


def _build_estimator_by_simulator_summary(
    metric_rows_df: pd.DataFrame,
    *,
    metric_col: str,
    rank: bool = False,
    sort_by_real_dgp: bool = False,
) -> pd.DataFrame:
    if metric_rows_df.empty:
        return pd.DataFrame()

    summary_df = (
        metric_rows_df.groupby(["generator", "estimator"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "avg_metric"})
    )
    if rank:
        summary_df["avg_rank"] = (
            summary_df.groupby("generator")["avg_metric"]
            .rank(method="average", ascending=True)
            .astype(float)
        )
        values_col = "avg_rank"
    else:
        values_col = "avg_metric"

    pivot_df = summary_df.pivot(index="estimator", columns="generator", values=values_col).copy()
    generators = [g for g in sorted(pd.Index(pivot_df.columns).astype(str).tolist()) if g != ""]
    if "REAL_DGP" in generators:
        generators = ["REAL_DGP"] + [g for g in generators if g != "REAL_DGP"]
    pivot_df = pivot_df.reindex(columns=generators)
    pivot_df.index.name = "estimator"

    # Keep the estimator identifier in the exported summary.  The previous
    # index-only CSV silently lost this key when ``index=False`` was used,
    # forcing downstream table code to rely on a hard-coded row order.
    pivot_df = pivot_df.reset_index()
    if sort_by_real_dgp and "REAL_DGP" in pivot_df.columns:
        # MSE tables should present estimators from lowest to highest MSE in
        # the true-DGP benchmark.  The estimator name is a deterministic
        # tie-breaker, so regenerated tables are stable across runs.
        pivot_df = pivot_df.sort_values(
            ["REAL_DGP", "estimator"], kind="stable", na_position="last"
        ).reset_index(drop=True)
    else:
        pivot_df = pivot_df.sort_values("estimator", kind="stable").reset_index(drop=True)
    return pivot_df


def main():
    args = _parse_args()
    bundle_path = Path(args.bundle)
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(bundle_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    promotion_rows = []
    distance_rows = []
    missing_runs = []
    reference_rows = []
    metric_rows = []
    ambiguous_estimands = []

    for run_name, payload in _iter_run_payloads(bundle_path):
        seed = _seed_from_name(run_name)
        try:
            perf_df, dist_df, embedded_benchmark_perf_df = _read_run_csvs(payload)
        except Exception as exc:
            missing_runs.append({"run_name": run_name, "seed": seed, "reason": str(exc)})
            continue

        if embedded_benchmark_perf_df is None:
            missing_runs.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "reason": "No embedded _dgp_benchmark MASTER_mc_summary.csv found.",
                }
            )
            continue
        benchmark_perf_df = embedded_benchmark_perf_df

        reference_estimand = _unique_estimand(benchmark_perf_df, "REAL_DGP", allow_mixed=args.allow_mixed_estimands)

        reference_rank_df = _build_real_dgp_rank_table(benchmark_perf_df)
        real_rank = dict(zip(reference_rank_df["estimator"], reference_rank_df["rank_mc_mse"]))
        real_mse = dict(zip(reference_rank_df["estimator"], reference_rank_df["mc_mse"]))
        for row in reference_rank_df.to_dict(orient="records"):
            row["seed"] = seed
            reference_rows.append(row)

        for generator in sorted(perf_df["generator"].astype(str).unique()):
            gen_df = perf_df.loc[perf_df["generator"].astype(str) == generator].copy()
            if gen_df.empty:
                continue
            simulator_estimand = _unique_estimand(perf_df, generator, allow_mixed=args.allow_mixed_estimands)
            if args.allow_mixed_estimands and not np.isfinite(simulator_estimand):
                ambiguous_estimands.append({'seed': seed, 'generator': generator,
                    'reason': 'Missing or mixed ate_true; estimand-gap analysis omitted; original metrics preserved'})

            for row in gen_df[["estimator", "mc_mse", "mc_abs_bias", "mc_bias", "mc_sd"]].to_dict(orient="records"):
                if "estimator" not in row or row["estimator"] is None:
                    continue
                metric_rows.append(
                    {
                        "seed": seed,
                        "generator": generator,
                        "estimator": str(row["estimator"]),
                        "mc_mse": float(row.get("mc_mse", np.nan)) if pd.notna(row.get("mc_mse", np.nan)) else np.nan,
                        "mc_abs_bias": float(row.get("mc_abs_bias", np.nan)) if pd.notna(row.get("mc_abs_bias", np.nan)) else np.nan,
                        "mc_bias": float(row.get("mc_bias", np.nan)) if pd.notna(row.get("mc_bias", np.nan)) else np.nan,
                        "mc_sd": float(row.get("mc_sd", np.nan)) if pd.notna(row.get("mc_sd", np.nan)) else np.nan,
                    }
                )

            if generator == "REAL_DGP":
                continue

            ranked = _rank_one_generator(perf_df, generator)
            if ranked.empty:
                continue

            sim_rank = dict(zip(ranked["estimator"], ranked["sim_rank"]))
            common = [e for e in ranked["estimator"] if e in real_rank]
            if not common:
                continue

            full_real = [real_rank[e] for e in common]
            full_sim = [sim_rank[e] for e in common]
            full_rho = float(spearmanr(full_real, full_sim).statistic)
            full_mard = float(np.mean([abs(sim_rank[e] - real_rank[e]) for e in common]))

            excluded = [e for e in common if not _is_family_aligned(generator, e)]
            excl_real = [real_rank[e] for e in excluded]
            excl_sim = [sim_rank[e] for e in excluded]
            excl_rho = float(spearmanr(excl_real, excl_sim).statistic) if len(excluded) >= 2 else np.nan
            excl_mard = float(np.mean([abs(sim_rank[e] - real_rank[e]) for e in excluded])) if excluded else np.nan

            per_seed_rows.append(
                {
                    "seed": seed,
                    "generator": generator,
                    "full_rho": full_rho,
                    "full_mean_abs_rank_diff": full_mard,
                    "excl_rho": excl_rho,
                    "excl_mean_abs_rank_diff": excl_mard,
                    "reference_estimand": reference_estimand,
                    "simulator_estimand": simulator_estimand,
                    "estimand_gap": simulator_estimand - reference_estimand,
                    "abs_estimand_gap": abs(simulator_estimand - reference_estimand),
                }
            )

            dist_all = dist_df.loc[
                (dist_df["generator"].astype(str) == generator)
                & (dist_df["group"].astype(str) == "ALL")
            ].copy()
            if not dist_all.empty:
                one = dist_all.iloc[0].to_dict()
                # Jensen--Shannon histogram distance is intentionally excluded
                # from the analysis outputs and paper tables.
                one.pop("JS_hist", None)
                one.pop("JS_hist_mean", None)
                one.update({"seed": seed, "generator": generator, "full_rho": full_rho, "excl_rho": excl_rho})
                distance_rows.append(one)

            for estimator in common:
                rank_diff = int(sim_rank[estimator] - real_rank[estimator])
                promotion_rows.append(
                    {
                        "seed": seed,
                        "generator": generator,
                        "estimator": estimator,
                        "real_rank": int(real_rank[estimator]),
                        "sim_rank": int(sim_rank[estimator]),
                        "rank_diff": rank_diff,
                        "promotion": -rank_diff,
                        "family_aligned": _is_family_aligned(generator, estimator),
                        "real_dgp_mc_mse": float(real_mse[estimator]),
                        "simulator_mc_mse": float(ranked.loc[ranked["estimator"] == estimator, "mc_mse"].iloc[0]),
                    }
                )

        if benchmark_perf_df is not None:
            real_dgp_rows = benchmark_perf_df.loc[
                benchmark_perf_df["generator"].astype(str) == "REAL_DGP",
                ["estimator", "mc_mse", "mc_abs_bias", "mc_bias", "mc_sd"],
            ].copy()
            for row in real_dgp_rows.to_dict(orient="records"):
                if "estimator" not in row or row["estimator"] is None:
                    continue
                metric_rows.append(
                    {
                        "seed": seed,
                        "generator": "REAL_DGP",
                        "estimator": str(row["estimator"]),
                        "mc_mse": float(row.get("mc_mse", np.nan)) if pd.notna(row.get("mc_mse", np.nan)) else np.nan,
                        "mc_abs_bias": float(row.get("mc_abs_bias", np.nan)) if pd.notna(row.get("mc_abs_bias", np.nan)) else np.nan,
                        "mc_bias": float(row.get("mc_bias", np.nan)) if pd.notna(row.get("mc_bias", np.nan)) else np.nan,
                        "mc_sd": float(row.get("mc_sd", np.nan)) if pd.notna(row.get("mc_sd", np.nan)) else np.nan,
                    }
                )

    per_seed_df = pd.DataFrame(
        per_seed_rows,
        columns=[
            "seed", "generator", "full_rho", "full_mean_abs_rank_diff", "excl_rho", "excl_mean_abs_rank_diff",
            "reference_estimand", "simulator_estimand", "estimand_gap", "abs_estimand_gap",
        ],
    )
    promotion_df = pd.DataFrame(
        promotion_rows,
        columns=["seed", "generator", "estimator", "real_rank", "sim_rank", "rank_diff", "promotion", "family_aligned", "real_dgp_mc_mse", "simulator_mc_mse"],
    )
    distance_run_df = pd.DataFrame(distance_rows)
    missing_df = pd.DataFrame(missing_runs)
    metric_rows_df = pd.DataFrame(
        metric_rows,
        columns=["seed", "generator", "estimator", "mc_mse", "mc_abs_bias", "mc_bias", "mc_sd"],
    )

    if per_seed_df.empty:
        avg_rho_df = pd.DataFrame(
            columns=["generator", "avg_full_rho", "sd_full_rho", "avg_excl_rho", "sd_excl_rho", "avg_full_mard", "avg_excl_mard"]
        )
        avg_full_rho_df = avg_rho_df[["generator", "avg_full_rho", "sd_full_rho", "avg_full_mard"]]
        avg_excl_rho_only_df = avg_rho_df[["generator", "avg_excl_rho", "sd_excl_rho", "avg_excl_mard"]]
        full_winners_df = pd.DataFrame(columns=["generator", "n_full_wins"])
        excl_winners_df = pd.DataFrame(columns=["generator", "n_excl_wins"])
        promotion_summary_df = pd.DataFrame(columns=["generator", "family_aligned", "mean_promotion", "mean_abs_rank_shift"])
        promotion_summary_excl_df = pd.DataFrame(columns=["generator", "mean_promotion", "mean_abs_rank_shift"])
        estimator_promotion_df = pd.DataFrame(columns=["generator", "estimator", "mean_promotion", "mean_rank_diff"])
    else:
        avg_rho_df = (
            per_seed_df.groupby("generator", as_index=False)
            .agg(
                avg_full_rho=("full_rho", "mean"),
                sd_full_rho=("full_rho", "std"),
                avg_excl_rho=("excl_rho", "mean"),
                sd_excl_rho=("excl_rho", "std"),
                avg_full_mard=("full_mean_abs_rank_diff", "mean"),
                avg_excl_mard=("excl_mean_abs_rank_diff", "mean"),
            )
            .sort_values("avg_full_rho", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        avg_full_rho_df = (
            avg_rho_df[["generator", "avg_full_rho", "sd_full_rho", "avg_full_mard"]]
            .sort_values("avg_full_rho", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        avg_excl_rho_only_df = (
            avg_rho_df[["generator", "avg_excl_rho", "sd_excl_rho", "avg_excl_mard"]]
            .sort_values("avg_excl_rho", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

        full_winners_df = (
            per_seed_df.loc[per_seed_df.groupby("seed")["full_rho"].idxmax()]
            .groupby("generator")
            .size()
            .reset_index(name="n_full_wins")
            .sort_values("n_full_wins", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        excl_winners_df = (
            per_seed_df.loc[per_seed_df.groupby("seed")["excl_rho"].idxmax()]
            .groupby("generator")
            .size()
            .reset_index(name="n_excl_wins")
            .sort_values("n_excl_wins", ascending=False, kind="stable")
            .reset_index(drop=True)
        )

        promotion_summary_df = (
            promotion_df.groupby(["generator", "family_aligned"], as_index=False)
            .agg(
                mean_promotion=("promotion", "mean"),
                mean_abs_rank_shift=("rank_diff", lambda s: float(np.mean(np.abs(np.asarray(s, dtype=float))))),
            )
            .sort_values(["generator", "family_aligned"], ascending=[True, False], kind="stable")
            .reset_index(drop=True)
        )
        promotion_summary_excl_df = (
            promotion_summary_df.loc[promotion_summary_df["family_aligned"] == False, ["generator", "mean_promotion", "mean_abs_rank_shift"]]
            .sort_values("mean_promotion", ascending=False, kind="stable")
            .reset_index(drop=True)
        )
        estimator_promotion_df = (
            promotion_df.groupby(["generator", "estimator"], as_index=False)
            .agg(
                mean_promotion=("promotion", "mean"),
                mean_rank_diff=("rank_diff", "mean"),
            )
            .sort_values(["mean_promotion", "generator", "estimator"], ascending=[False, True, True], kind="stable")
            .reset_index(drop=True)
        )

    estimand_gap_corr_df = _estimand_gap_correlation_table(per_seed_df)

    if distance_run_df.empty or "generator" not in distance_run_df.columns:
        dist_corr_full_no_wgan_df = pd.DataFrame(columns=["distance_metric", "pearson_with_rho", "spearman_with_rho", "n_pairs"])
        dist_corr_excl_no_wgan_df = pd.DataFrame(columns=["distance_metric", "pearson_with_rho", "spearman_with_rho", "n_pairs"])
        dist_corr_full_all_df = pd.DataFrame(columns=["distance_metric", "pearson_with_rho", "spearman_with_rho", "n_pairs"])
        dist_corr_excl_all_df = pd.DataFrame(columns=["distance_metric", "pearson_with_rho", "spearman_with_rho", "n_pairs"])
    else:
        dist_corr_full_no_wgan_df = _distance_rho_correlation_table(
            distance_run_df.loc[distance_run_df["generator"].astype(str) != "WGAN"].copy(),
            rho_col="full_rho",
        )
        dist_corr_excl_no_wgan_df = _distance_rho_correlation_table(
            distance_run_df.loc[distance_run_df["generator"].astype(str) != "WGAN"].copy(),
            rho_col="excl_rho",
        )

        if args.include_wgan_in_distance_corr:
            dist_corr_full_all_df = _distance_rho_correlation_table(distance_run_df.copy(), rho_col="full_rho")
            dist_corr_excl_all_df = _distance_rho_correlation_table(distance_run_df.copy(), rho_col="excl_rho")
        else:
            dist_corr_full_all_df = pd.DataFrame(columns=["distance_metric", "pearson_with_rho", "spearman_with_rho", "n_pairs"])
            dist_corr_excl_all_df = pd.DataFrame(columns=["distance_metric", "pearson_with_rho", "spearman_with_rho", "n_pairs"])

    reference_df = pd.DataFrame(reference_rows)
    if not reference_df.empty:
        reference_df = reference_df.sort_values(["seed", "rank_mc_mse", "estimator"], kind="stable").reset_index(drop=True)
    reference_df.to_csv(out_dir / "reference_real_dgp_ranking.csv", index=False)

    mse_summary_df = _build_estimator_by_simulator_summary(
        metric_rows_df, metric_col="mc_mse", sort_by_real_dgp=True
    )
    bias_summary_df = _build_estimator_by_simulator_summary(metric_rows_df, metric_col="mc_bias")
    abs_bias_summary_df = _build_estimator_by_simulator_summary(metric_rows_df, metric_col="mc_abs_bias")
    sd_summary_df = _build_estimator_by_simulator_summary(metric_rows_df, metric_col="mc_sd")
    mse_rank_summary_df = _build_estimator_by_simulator_summary(metric_rows_df, metric_col="mc_mse", rank=True)

    mse_summary_df.to_csv(out_dir / "average_mc_mse_by_estimator_and_simulator.csv", index=False)
    bias_summary_df.to_csv(out_dir / "average_mc_bias_by_estimator_and_simulator.csv", index=False)
    abs_bias_summary_df.to_csv(out_dir / "average_mc_abs_bias_by_estimator_and_simulator.csv", index=False)
    sd_summary_df.to_csv(out_dir / "average_mc_sd_by_estimator_and_simulator.csv", index=False)
    mse_rank_summary_df.to_csv(out_dir / "average_rank_mc_mse_by_estimator_and_simulator.csv", index=False)
    per_seed_df.sort_values(["generator", "seed"], kind="stable").to_csv(out_dir / "per_seed_rho.csv", index=False)
    estimand_gap_corr_df.to_csv(out_dir / "estimand_gap_correlation.csv", index=False)
    avg_rho_df.to_csv(out_dir / "average_rho_by_generator.csv", index=False)
    full_winners_df.to_csv(out_dir / "full_rho_winner_counts.csv", index=False)
    excl_winners_df.to_csv(out_dir / "excluded_family_rho_winner_counts.csv", index=False)
    promotion_summary_df.to_csv(out_dir / "promotion_summary.csv", index=False)
    estimator_promotion_df.to_csv(out_dir / "estimator_promotion_summary.csv", index=False)
    promotion_df.to_csv(out_dir / "promotion_detail.csv", index=False)
    gcomp_selection_summary = write_gcomp_selection_comparison(promotion_df, out_dir)
    gcomp_alignment_by_seed, gcomp_alignment_summary = gcomp_only_alignment_metrics(promotion_df)
    gcomp_alignment_by_seed.to_csv(out_dir / "gcomp_only_alignment_by_seed.csv", index=False)
    gcomp_alignment_summary.to_csv(out_dir / "gcomp_only_alignment_summary.csv", index=False)
    distance_run_df.to_csv(out_dir / "distance_run_level.csv", index=False)
    dist_corr_full_no_wgan_df.to_csv(out_dir / "distance_correlation_full_rho_no_wgan.csv", index=False)
    dist_corr_excl_no_wgan_df.to_csv(out_dir / "distance_correlation_excluded_rho_no_wgan.csv", index=False)
    if args.include_wgan_in_distance_corr:
        dist_corr_full_all_df.to_csv(out_dir / "distance_correlation_full_rho_all.csv", index=False)
        dist_corr_excl_all_df.to_csv(out_dir / "distance_correlation_excluded_rho_all.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_dir / "missing_or_unusable_runs.csv", index=False)

    lines = []
    lines.append(f"Bundle: {bundle_path}")
    lines.append("Benchmark source: embedded per-run REAL_DGP benchmark")
    lines.append(f"Usable seeds: {per_seed_df['seed'].nunique()}")
    lines.append(f"Missing/unusable runs: {len(missing_df)}")
    lines.append(f"Ambiguous simulator estimands omitted from gap analysis: {len(ambiguous_estimands)}")
    pd.DataFrame(ambiguous_estimands, columns=['seed', 'generator', 'reason']).to_csv(
        out_dir / 'ambiguous_estimands.csv', index=False)
    lines.append("")
    lines.append("Average rho by generator (all estimators)")
    lines.append(avg_full_rho_df.to_string(index=False))
    lines.append("")
    lines.append("Average rho by generator (excluding own-family estimators)")
    lines.append(avg_excl_rho_only_df.to_string(index=False))
    lines.append("")
    lines.append("Full-rho winner counts")
    lines.append(full_winners_df.to_string(index=False))
    lines.append("")
    lines.append("Excluded-family-rho winner counts")
    lines.append(excl_winners_df.to_string(index=False))
    lines.append("")
    lines.append("Average metrics by estimator and simulator")
    lines.append("Average mc_mse")
    lines.append(mse_summary_df.to_string())
    lines.append("")
    lines.append("Average mc_bias")
    lines.append(bias_summary_df.to_string())
    lines.append("")
    lines.append("Average mc_abs_bias")
    lines.append(abs_bias_summary_df.to_string())
    lines.append("")
    lines.append("Average mc_sd")
    lines.append(sd_summary_df.to_string())
    lines.append("")
    lines.append("Average rank of estimators by mc_mse")
    lines.append(mse_rank_summary_df.to_string())
    lines.append("")
    lines.append("Average metrics by estimator and simulator (CSV files also written)")
    lines.append("- average_mc_mse_by_estimator_and_simulator.csv")
    lines.append("- average_mc_bias_by_estimator_and_simulator.csv")
    lines.append("- average_mc_abs_bias_by_estimator_and_simulator.csv")
    lines.append("- average_mc_sd_by_estimator_and_simulator.csv")
    lines.append("- average_rank_mc_mse_by_estimator_and_simulator.csv")
    lines.append("")
    lines.append("Promotion summary (all estimators)")
    lines.append(promotion_summary_df.to_string(index=False))
    lines.append("")
    lines.append("Promotion summary (excluding own-family estimators)")
    lines.append(promotion_summary_excl_df.to_string(index=False))
    lines.append("")
    lines.append("G-comp-only alignment diagnostics (only associated G-comp marked aligned)")
    lines.append(gcomp_alignment_summary.to_string(index=False))
    lines.append("")
    lines.append("Associated G-comp versus simulator-selected estimator (target MSE)")
    lines.append("All: select from all estimators. Excl: select from non-aligned estimators; G-comp remains the baseline.")
    lines.append("Simulator-MSE ties: estimator-name order. Target-MSE ties: rtol=1e-12, atol=1e-15.")
    lines.append("Regret subtracts the best target MSE over all estimators, using the same comparator in All and Excl.")
    lines.append("Relative regret (%) = 100 * (chosen target MSE / best target MSE - 1), computed per seed then averaged.")
    lines.append("Zero best MSE: relative regret undefined; n_relative_regret reports the usable denominator.")
    lines.append(gcomp_selection_summary.to_string(index=False))
    lines.append("")
    lines.append("Association of absolute estimand gap with rank correlation")
    lines.append(estimand_gap_corr_df.to_string(index=False))
    lines.append("")
    lines.append("Distance correlations with full rho (no WGAN)")
    lines.append(dist_corr_full_no_wgan_df.to_string(index=False))
    lines.append("")
    lines.append("Distance correlations with excluded-family rho (no WGAN)")
    lines.append(dist_corr_excl_no_wgan_df.to_string(index=False))
    if args.include_wgan_in_distance_corr:
        lines.append("")
        lines.append("Distance correlations with full rho (all generators)")
        lines.append(dist_corr_full_all_df.to_string(index=False))
        lines.append("")
        lines.append("Distance correlations with excluded-family rho (all generators)")
        lines.append(dist_corr_excl_all_df.to_string(index=False))
    (out_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Analysis written to: {out_dir}")
    print(f"Usable seeds: {per_seed_df['seed'].nunique()}")
    print(f"Missing/unusable runs: {len(missing_df)}")
    print("")
    print(avg_rho_df.to_string(index=False))


if __name__ == "__main__":
    main()
