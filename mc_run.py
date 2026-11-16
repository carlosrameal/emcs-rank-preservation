"""Run Monte Carlo replications and summarize estimator performance."""

import os
import queue
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import inspect
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance
from tqdm.auto import tqdm

from mc_workers import do_rep_chunk, init_chunk_worker

# ==============================================================================
# SMALL HELPERS
# ==============================================================================

_GENERATOR_OWN_FAMILY_ESTIMATORS: dict[str, set[str]] = {
    "OLS": {"OLS1", "OLS2", "OLS2_GCOMP", "GB_OLS2_AIPW", "GB_OLS2_TMLE"},
    "GB": {"GB_GCOMP", "GB_GB_AIPW", "GB_GB_TMLE"},
    "RF": {"RF_GCOMP", "GB_RF_AIPW", "GB_RF_TMLE"},
    "NN": {"NN_GCOMP", "GB_NN_AIPW", "GB_NN_TMLE"},
    "WGAN": set(),
    "REAL_DGP": set(),
}

def _safe_float(x):
    """
    Convert x to float when possible, else np.nan.
    Handles None gracefully.
    """
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _safe_bool_to_float(x):
    """
    Convert booleans / truthy values to 0/1 float, else np.nan.
    """
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        try:
            return float(bool(x))
        except Exception:
            return np.nan


def _default_estimator_class_map(perf_df: pd.DataFrame) -> dict:
    """
    Fallback grouping by estimator family when no explicit map is provided.
    Adjust naming rules if needed for your exact estimator names.
    """
    ests = perf_df["estimator"].astype(str).unique().tolist()
    out = {}

    for e in ests:
        eu = e.upper()

        if eu in {"OLS1", "OLS2", "OLS2_GCOMP", "ORACLE_OLS"} or "OLS" in eu:
            out[e] = "OLS BASELINES"
        elif "IPW" in eu and "AIPW" not in eu:
            out[e] = "IPW"
        elif "GCOMP" in eu:
            out[e] = "G-COMP"
        elif "AIPW" in eu:
            out[e] = "AIPW"
        elif "BART" in eu or "XBART" in eu:
            out[e] = "BART / XBART"
        else:
            out[e] = "OTHER"

    return out


def _format_numeric_df_for_report(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    """
    Return a copy with numeric columns rounded for prettier text output.
    Leaves non-numeric columns unchanged.
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


def available_cpus() -> int:
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k)
        if v is not None:
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                pass

    try:
        if hasattr(os, "sched_getaffinity"):
            return int(len(os.sched_getaffinity(0)))
    except Exception:
        pass

    return int(max(1, os.cpu_count() or 1))


def js_divergence_hist(x, y, bins="fd"):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    hist_x, bin_edges = np.histogram(x, bins=bins, density=True)
    hist_y, _ = np.histogram(y, bins=bin_edges, density=True)

    px = hist_x / (hist_x.sum() + 1e-12)
    py = hist_y / (hist_y.sum() + 1e-12)
    return float(jensenshannon(px, py, base=2.0) ** 2)


def distribution_distances(y_gen, y_ref, A_np, label):
    y_gen = np.asarray(y_gen).ravel()
    y_ref = np.asarray(y_ref).ravel()
    A_np = np.asarray(A_np).ravel()
    if not (len(y_gen) == len(y_ref) == len(A_np)):
        raise ValueError(f"Length mismatch: y_gen={len(y_gen)}, y_ref={len(y_ref)}, A={len(A_np)}")

    rows = []
    for grp_name, mask in [
        ("A=0", A_np == 0),
        ("A=1", A_np == 1),
        ("ALL", np.ones_like(A_np, dtype=bool)),
    ]:
        yg = y_gen[mask]
        yr = y_ref[mask]

        rows.append({
            "generator": str(label),
            "group": str(grp_name),
            "W1": float(wasserstein_distance(yr, yg)),
            "KS": float(ks_2samp(yr, yg).statistic),
            "Energy": float(energy_distance(yr, yg)),
            "JS_hist": float(js_divergence_hist(yr, yg, bins="fd")),
            "MeanDiff": float(np.mean(yg) - np.mean(yr)),
            "StdRatio": float(np.std(yg, ddof=1) / (np.std(yr, ddof=1) + 1e-12)),
        })
    return rows


def _filter_kwargs_for_fn(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        return {}


# ==============================================================================
# CHUNKED PROCESS-BASED RUN FOR ONE GENERATOR
# ==============================================================================


def _resolve_mc_process_context():
    preferred = ("fork", "spawn") if os.name != "nt" else ("spawn",)
    for method in preferred:
        try:
            return mp.get_context(method), method
        except ValueError:
            continue
    return None, "default"


def _split_rep_indices(n_mc: int, n_workers: int) -> list[list[int]]:
    rep_indices = np.arange(int(n_mc), dtype=int)
    splits = np.array_split(rep_indices, max(1, min(int(n_workers), int(n_mc))))
    return [chunk.astype(int).tolist() for chunk in splits if len(chunk)]


def _drain_progress_queue(progress_queue, completed: int) -> int:
    total = int(completed)
    while True:
        try:
            total += int(progress_queue.get_nowait())
        except queue.Empty:
            break
    return total


def _format_progress_stats(completed: int, total: int, elapsed: float) -> str:
    completed = int(max(0, completed))
    total = int(max(1, total))
    elapsed = float(max(0.0, elapsed))
    if completed <= 0 or elapsed <= 0.0:
        return ""
    reps_per_sec = completed / elapsed
    secs_per_rep = elapsed / completed
    remaining = max(0, total - completed)
    eta = (remaining / reps_per_sec) if reps_per_sec > 0 else float("nan")
    return (
        f" | rate={reps_per_sec:.3f} reps/s"
        f" | sec/rep={secs_per_rep:.3f}"
        f" | eta={eta:.2f}s"
    )


def _summarize_distance_rows(dist_rows: list[dict]) -> pd.DataFrame:
    if not dist_rows:
        return pd.DataFrame()

    dist_df = pd.DataFrame(dist_rows)
    if dist_df.empty or not {"generator", "group"}.issubset(dist_df.columns):
        return pd.DataFrame()

    metric_candidates = ["W1", "KS", "Energy", "JS_hist", "MeanDiff", "StdRatio"]
    metrics = [c for c in metric_candidates if c in dist_df.columns]
    if not metrics:
        return pd.DataFrame()

    return (
        dist_df
        .groupby(["generator", "group"], as_index=False)
        .agg(
            **{f"{m}_mean": (m, "mean") for m in metrics},
            **{f"{m}_sd": (m, "std") for m in metrics},
        )
    )


def _prepare_observed_draws(
    *,
    gen,
    gen_label: str,
    A_np: np.ndarray,
    n_mc: int,
    seed_base: int,
):
    print(
        f"[{gen_label}] pre-generating {int(n_mc)} observed synthetic datasets in main process...",
        flush=True,
    )
    t0 = time.perf_counter()

    a_draws = None

    if hasattr(gen, "draw_observed"):
        y_obs_draws, a_draws = gen.draw_observed(seed=int(seed_base), n_samples=int(n_mc))
        y_obs_draws = np.asarray(y_obs_draws, dtype=np.float32)
        a_draws = np.asarray(a_draws, dtype=np.int8)
        if y_obs_draws.ndim == 1:
            y_obs_draws = y_obs_draws[:, None]
        if a_draws.ndim == 1:
            a_draws = a_draws[:, None]
    elif hasattr(gen, "draw_y0") and hasattr(gen, "draw_y1"):
        y0_draws = np.asarray(gen.draw_y0(seed=int(seed_base), n_samples=int(n_mc)), dtype=np.float32)
        y1_draws = np.asarray(
            gen.draw_y1(seed=int(seed_base) + 1_000_000, n_samples=int(n_mc)),
            dtype=np.float32,
        )
        if y0_draws.ndim == 1:
            y0_draws = y0_draws[:, None]
        if y1_draws.ndim == 1:
            y1_draws = y1_draws[:, None]
        a_col = np.asarray(A_np).reshape(-1, 1).astype(int)
        y_obs_draws = np.where(a_col == 1, y1_draws, y0_draws).astype(np.float32, copy=False)
    else:
        cols = []
        for r in range(int(n_mc)):
            seed = int(seed_base) + int(r)
            cols.append(np.asarray(gen(seed), dtype=np.float32).reshape(-1))
        y_obs_draws = np.column_stack(cols).astype(np.float32, copy=False)

    print(
        f"[{gen_label}] finished pre-generating observed datasets in "
        f"{time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return y_obs_draws, a_draws

def run_for_generator(
    *,
    gen,
    gen_label: str,
    y_ref: np.ndarray,
    true_ate: float,
    estimators: dict,
    estimator_params: dict,
    X_df: pd.DataFrame,
    A_np: np.ndarray,
    n_mc: int,
    seed_base: int,
    alpha: float,
    n_workers: int,
    threads_per_worker: int,
):
    y_draws, a_draws = _prepare_observed_draws(
        gen=gen,
        gen_label=gen_label,
        A_np=A_np,
        n_mc=n_mc,
        seed_base=seed_base,
    )

    outer_workers = max(1, min(int(n_workers), int(n_mc)))
    estimator_threads = max(1, int(threads_per_worker))
    interactive_progress = bool(getattr(sys.stdout, "isatty", lambda: False)())
    progress_step = max(1, min(50, int(n_mc)))
    rep_chunks = _split_rep_indices(n_mc=int(n_mc), n_workers=outer_workers)
    chunk_sizes = [len(chunk) for chunk in rep_chunks]
    process_ctx, process_method = _resolve_mc_process_context()

    print(
        f"[{gen_label}] chunk plan: {len(rep_chunks)} worker processes "
        f"| chunk_sizes={chunk_sizes} | process_start={process_method}",
        flush=True,
    )

    out = []
    progress_t0 = time.perf_counter()
    executor_kwargs = {"max_workers": outer_workers}
    if process_ctx is not None:
        executor_kwargs["mp_context"] = process_ctx

    progress_manager = mp.Manager()
    progress_queue = progress_manager.Queue()
    progress_completed = 0
    progress_reported = 0

    try:
        with ProcessPoolExecutor(
            initializer=init_chunk_worker,
            initargs=(
                gen_label,
                y_ref,
                true_ate,
                estimators,
                estimator_params,
                seed_base,
                alpha,
                estimator_threads,
                X_df,
                A_np,
                a_draws,
                y_draws,
                progress_queue,
                progress_step,
            ),
            **executor_kwargs,
        ) as ex:
            pending = {
                ex.submit(do_rep_chunk, chunk): tuple(chunk)
                for chunk in rep_chunks
            }

            if interactive_progress:
                with tqdm(
                    total=int(n_mc),
                    desc=f"MC reps [{gen_label}]",
                    unit="rep",
                    dynamic_ncols=True,
                    file=sys.stdout,
                ) as pbar:
                    while pending:
                        done, not_done = wait(
                            pending.keys(),
                            timeout=1.0,
                            return_when=FIRST_COMPLETED,
                        )
                        pending = {fut: pending[fut] for fut in not_done}

                        new_completed = _drain_progress_queue(progress_queue, progress_completed)
                        delta = new_completed - progress_completed
                        if delta > 0:
                            progress_completed = new_completed
                            pbar.update(int(delta))

                        for fut in done:
                            out.append(fut.result())

                    new_completed = _drain_progress_queue(progress_queue, progress_completed)
                    delta = new_completed - progress_completed
                    if delta > 0:
                        progress_completed = new_completed
                        pbar.update(int(delta))
                    if pbar.n < int(n_mc):
                        pbar.update(int(n_mc) - pbar.n)
            else:
                while pending:
                    done, not_done = wait(
                        pending.keys(),
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )
                    pending = {fut: pending[fut] for fut in not_done}

                    progress_completed = _drain_progress_queue(progress_queue, progress_completed)
                    while progress_reported + progress_step <= progress_completed:
                        progress_reported += progress_step
                        elapsed = time.perf_counter() - progress_t0
                        print(
                            f"MC reps [{gen_label}]: {progress_reported}/{int(n_mc)} complete "
                            f"| elapsed={elapsed:.2f}s"
                            f"{_format_progress_stats(progress_reported, int(n_mc), elapsed)}",
                            flush=True,
                        )

                    for fut in done:
                        out.append(fut.result())

                progress_completed = _drain_progress_queue(progress_queue, progress_completed)
                if progress_completed < int(n_mc):
                    progress_completed = int(n_mc)
                if progress_reported < progress_completed:
                    elapsed = time.perf_counter() - progress_t0
                    print(
                        f"MC reps [{gen_label}]: {progress_completed}/{int(n_mc)} complete "
                        f"| elapsed={elapsed:.2f}s"
                        f"{_format_progress_stats(progress_completed, int(n_mc), elapsed)}",
                        flush=True,
                    )
    finally:
        progress_manager.shutdown()

    est_rows = []
    dist_rows = []

    for item in out:
        est_rows.extend(item["est_rows"])
        dist_rows.extend(item["dist_rows"])

    est_df = pd.DataFrame(
        est_rows,
        columns=["r", "seed", "estimator", "ate_hat", "se_hat", "ci_lo", "ci_hi", "cover"],
    )

    # Make downstream summaries robust to None / mixed types
    if not est_df.empty:
        est_df["ate_hat"] = est_df["ate_hat"].map(_safe_float)
        est_df["se_hat"] = est_df["se_hat"].map(_safe_float)
        est_df["ci_lo"] = est_df["ci_lo"].map(_safe_float)
        est_df["ci_hi"] = est_df["ci_hi"].map(_safe_float)
        est_df["cover"] = est_df["cover"].map(_safe_bool_to_float)

    dist_df = _summarize_distance_rows(dist_rows)

    return est_df, dist_df


# ==============================================================================
# SUMMARY METRICS
# ==============================================================================

def summarize_mc(est_df: pd.DataFrame, true_ate: float) -> pd.DataFrame:
    """
    Summarize Monte Carlo performance by estimator.

    Robust to:
      - se_hat being None / missing
      - ci_lo / ci_hi being None / missing
      - cover being None / missing

    Reported columns:
      - ate_true
      - mc_mean_ate
      - mc_bias
      - mc_abs_bias
      - mc_sd
      - mc_var
      - mc_mse
      - rmse
      - mean_se_hat            (NaN if unavailable)
      - se_bias                (mean_se_hat - mc_sd, NaN if unavailable)
      - mean_ci_len            (NaN if unavailable)
      - coverage_rate          (NaN if unavailable)
      - n_valid_se
      - n_valid_ci
      - n_valid_cover
      - N_MC
    """
    out = []

    if est_df.empty:
        return pd.DataFrame(
            columns=[
                "estimator",
                "ate_true",
                "mc_mean_ate",
                "mc_bias",
                "mc_abs_bias",
                "mc_sd",
                "mc_var",
                "mc_mse",
                "rmse",
                "mean_se_hat",
                "se_bias",
                "mean_ci_len",
                "coverage_rate",
                "n_valid_se",
                "n_valid_ci",
                "n_valid_cover",
                "N_MC",
            ]
        )

    for est_name, g in est_df.groupby("estimator", dropna=False):
        ate_hat = pd.to_numeric(g["ate_hat"], errors="coerce").to_numpy(dtype=float)
        se_hat = pd.to_numeric(g["se_hat"], errors="coerce").to_numpy(dtype=float)
        ci_lo = pd.to_numeric(g["ci_lo"], errors="coerce").to_numpy(dtype=float)
        ci_hi = pd.to_numeric(g["ci_hi"], errors="coerce").to_numpy(dtype=float)
        cover = pd.to_numeric(g["cover"], errors="coerce").to_numpy(dtype=float)

        n_mc = int(len(g))

        mc_mean = float(np.nanmean(ate_hat)) if np.isfinite(ate_hat).any() else np.nan
        mc_bias = float(mc_mean - true_ate) if np.isfinite(mc_mean) else np.nan
        mc_abs_bias = float(abs(mc_bias)) if np.isfinite(mc_bias) else np.nan

        if np.isfinite(ate_hat).sum() >= 2:
            mc_sd = float(np.nanstd(ate_hat, ddof=1))
            mc_var = float(mc_sd ** 2)
        elif np.isfinite(ate_hat).sum() == 1:
            mc_sd = np.nan
            mc_var = np.nan
        else:
            mc_sd = np.nan
            mc_var = np.nan

        mc_mse = (
            float(np.nanmean((ate_hat - true_ate) ** 2))
            if np.isfinite(ate_hat).any()
            else np.nan
        )
        rmse = float(np.sqrt(mc_mse)) if np.isfinite(mc_mse) else np.nan

        valid_se = np.isfinite(se_hat)
        mean_se_hat = float(np.nanmean(se_hat)) if valid_se.any() else np.nan
        se_bias = (
            float(mean_se_hat - mc_sd)
            if np.isfinite(mean_se_hat) and np.isfinite(mc_sd)
            else np.nan
        )

        ci_len = ci_hi - ci_lo
        valid_ci = np.isfinite(ci_lo) & np.isfinite(ci_hi)
        mean_ci_len = float(np.nanmean(ci_len)) if valid_ci.any() else np.nan

        valid_cover = np.isfinite(cover)
        coverage_rate = float(np.nanmean(cover)) if valid_cover.any() else np.nan

        out.append(
            dict(
                estimator=str(est_name),
                ate_true=float(true_ate),
                mc_mean_ate=mc_mean,
                mc_bias=mc_bias,
                mc_abs_bias=mc_abs_bias,
                mc_sd=mc_sd,
                mc_var=mc_var,
                mc_mse=mc_mse,
                rmse=rmse,
                mean_se_hat=mean_se_hat,
                se_bias=se_bias,
                mean_ci_len=mean_ci_len,
                coverage_rate=coverage_rate,
                n_valid_se=int(valid_se.sum()),
                n_valid_ci=int(valid_ci.sum()),
                n_valid_cover=int(valid_cover.sum()),
                N_MC=n_mc,
            )
        )

    return pd.DataFrame(out)


# ==============================================================================
# REPORT FORMATTERS
# ==============================================================================

def _format_grouped_performance_table(
    perf_df: pd.DataFrame,
    estimator_class_map: dict | None = None,
) -> list[str]:
    """
    Build performance section grouped by estimator class, with all metrics
    shown per generator and per estimator.
    """
    if perf_df.empty:
        return ["(none)"]

    df = perf_df.copy()

    if estimator_class_map is None:
        estimator_class_map = _default_estimator_class_map(df)

    df["estimator_class"] = df["estimator"].map(estimator_class_map).fillna("OTHER")

    class_order = [
        "OLS BASELINES",
        "IPW",
        "G-COMP",
        "AIPW",
        "BART / XBART",
        "OTHER",
    ]
    present = [c for c in class_order if c in df["estimator_class"].unique()]
    extras = sorted([c for c in df["estimator_class"].unique() if c not in present])
    class_order = present + extras

    display_cols = [
        "estimator",
        "generator",
        "ate_true",
        "mc_mean_ate",
        "mc_bias",
        "mc_abs_bias",
        "mc_sd",
        "mc_var",
        "mc_mse",
        "rmse",
        "mean_se_hat",
        "se_bias",
        "mean_ci_len",
        "coverage_rate",
        "n_valid_se",
        "n_valid_ci",
        "n_valid_cover",
        "N_MC",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    lines = []
    for cls in class_order:
        g = df.loc[df["estimator_class"] == cls, display_cols].copy()
        if g.empty:
            continue

        g = g.sort_values(["estimator", "generator"], kind="stable")
        g = _format_numeric_df_for_report(g, digits=6)

        lines.append(f"[{cls}]")
        lines.append(g.to_string(index=False))
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    return lines


def _format_rankings_by_generator(perf_df: pd.DataFrame) -> list[str]:
    """
    Rankings within each generator.
    Main ranking criterion: lower mc_mse is better.
    Tie-breakers: lower abs bias, lower mc_sd, estimator name.
    """
    if perf_df.empty:
        return ["(none)"]

    lines = []
    generators = sorted(perf_df["generator"].astype(str).unique())

    for gen in generators:
        g = perf_df.loc[perf_df["generator"].astype(str) == gen].copy()

        # rank by main performance criterion
        g = g.sort_values(
            ["mc_mse", "mc_abs_bias", "mc_sd", "estimator"],
            ascending=[True, True, True, True],
            kind="stable",
        ).reset_index(drop=True)

        g["rank_mc_mse"] = np.arange(1, len(g) + 1)

        cols = [
            "rank_mc_mse",
            "estimator",
            "mc_mse",
            "rmse",
            "mc_abs_bias",
            "mc_bias",
            "mc_sd",
            "mc_mean_ate",
            "ate_true",
            "coverage_rate",
            "mean_ci_len",
        ]
        cols = [c for c in cols if c in g.columns]

        g = _format_numeric_df_for_report(g[cols], digits=6)

        lines.append(f"[{gen}]")
        lines.append(g.to_string(index=False))
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    return lines


def _rank_generator_df(perf_df: pd.DataFrame, *, generator: str) -> pd.DataFrame:
    g = perf_df.loc[perf_df["generator"].astype(str) == str(generator)].copy()
    if g.empty:
        return g

    g = g.sort_values(
        ["mc_mse", "mc_abs_bias", "mc_sd", "estimator"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    g["rank_mc_mse"] = np.arange(1, len(g) + 1)
    return g


def _spearman_rho_from_ranks(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        return float("nan")
    d2 = np.sum((x - y) ** 2)
    return float(1.0 - (6.0 * d2) / (n * (n**2 - 1.0)))


def _rerank_within_subset(
    ranked_df: pd.DataFrame,
    *,
    generator: str,
    excluded_estimators: set[str],
    rank_col_name: str,
    mse_col_name: str,
) -> pd.DataFrame:
    subset = ranked_df.loc[
        ~ranked_df["estimator"].astype(str).isin(set(excluded_estimators))
    ].copy()
    if subset.empty:
        return subset
    subset = subset.reset_index(drop=True)
    subset["rank_mc_mse"] = np.arange(1, len(subset) + 1)
    return subset[["estimator", "rank_mc_mse", "mc_mse"]].rename(
        columns={
            "rank_mc_mse": rank_col_name,
            "mc_mse": mse_col_name,
        }
    )


def _format_rank_comparison_vs_real_dgp(
    perf_df: pd.DataFrame,
    real_dgp_perf_df: pd.DataFrame | None,
) -> list[str]:
    if perf_df.empty or real_dgp_perf_df is None or real_dgp_perf_df.empty:
        return ["(none)"]

    real_ranked = _rank_generator_df(real_dgp_perf_df, generator="REAL_DGP")
    if real_ranked.empty:
        return ["(REAL_DGP rows missing from reference benchmark)"]

    real_cols = ["estimator", "rank_mc_mse", "mc_mse"]
    real_ranked_full = real_ranked[real_cols].rename(
        columns={
            "rank_mc_mse": "real_dgp_rank",
            "mc_mse": "real_dgp_mc_mse",
        }
    )

    lines: list[str] = []
    summary_rows = []
    generators = sorted(
        g for g in perf_df["generator"].astype(str).unique().tolist()
        if g != "REAL_DGP"
    )

    for gen in generators:
        sim_ranked = _rank_generator_df(perf_df, generator=gen)
        if sim_ranked.empty:
            continue

        sim_ranked_full = sim_ranked[["estimator", "rank_mc_mse", "mc_mse"]].rename(
            columns={
                "rank_mc_mse": "simulator_rank",
                "mc_mse": "simulator_mc_mse",
            }
        )

        merged = real_ranked_full.merge(sim_ranked_full, on="estimator", how="inner")
        if merged.empty:
            continue

        merged["rank_diff"] = merged["simulator_rank"] - merged["real_dgp_rank"]
        rho = _spearman_rho_from_ranks(
            merged["real_dgp_rank"].to_numpy(),
            merged["simulator_rank"].to_numpy(),
        )
        own_family_estimators = _GENERATOR_OWN_FAMILY_ESTIMATORS.get(str(gen), set())
        real_ranked_excl = _rerank_within_subset(
            real_ranked,
            generator="REAL_DGP",
            excluded_estimators=own_family_estimators,
            rank_col_name="real_dgp_rank",
            mse_col_name="real_dgp_mc_mse",
        )
        sim_ranked_excl = _rerank_within_subset(
            sim_ranked,
            generator=str(gen),
            excluded_estimators=own_family_estimators,
            rank_col_name="simulator_rank",
            mse_col_name="simulator_mc_mse",
        )
        merged_excl = real_ranked_excl.merge(sim_ranked_excl, on="estimator", how="inner")
        if not merged_excl.empty:
            merged_excl["rank_diff"] = merged_excl["simulator_rank"] - merged_excl["real_dgp_rank"]
        if len(merged_excl) >= 2:
            rho_excl = _spearman_rho_from_ranks(
                merged_excl["real_dgp_rank"].to_numpy(),
                merged_excl["simulator_rank"].to_numpy(),
            )
            mean_abs_rank_diff_excl = float(np.mean(np.abs(merged_excl["rank_diff"])))
            max_abs_rank_diff_excl = int(np.max(np.abs(merged_excl["rank_diff"])))
        else:
            rho_excl = float("nan")
            mean_abs_rank_diff_excl = float("nan")
            max_abs_rank_diff_excl = np.nan
        summary_rows.append(
            {
                "generator": gen,
                "n_common_estimators": int(len(merged)),
                "spearman_rho": float(rho),
                "mean_abs_rank_diff": float(np.mean(np.abs(merged["rank_diff"]))),
                "max_abs_rank_diff": int(np.max(np.abs(merged["rank_diff"]))),
                "n_excl_own_family": int(len(merged_excl)),
                "spearman_rho_excl_own_family": float(rho_excl),
                "mean_abs_rank_diff_excl_own_family": mean_abs_rank_diff_excl,
                "max_abs_rank_diff_excl_own_family": max_abs_rank_diff_excl,
            }
        )

        display = merged.sort_values("real_dgp_rank", kind="stable").reset_index(drop=True)
        display = _format_numeric_df_for_report(
            display[
                [
                    "estimator",
                    "real_dgp_rank",
                    "simulator_rank",
                    "rank_diff",
                    "real_dgp_mc_mse",
                    "simulator_mc_mse",
                ]
            ],
            digits=6,
        )
        lines.append(f"[{gen}]")
        lines.append(f"Spearman rho vs REAL_DGP ranks: {rho:.6f}")
        if len(merged_excl) >= 2:
            lines.append(
                "Spearman rho vs REAL_DGP ranks "
                f"(excluding own-family estimators): {rho_excl:.6f}"
            )
        else:
            lines.append(
                "Spearman rho vs REAL_DGP ranks "
                "(excluding own-family estimators): nan"
            )
        lines.append(display.to_string(index=False))
        if not merged_excl.empty:
            display_excl = merged_excl.sort_values("real_dgp_rank", kind="stable").reset_index(drop=True)
            display_excl = _format_numeric_df_for_report(
                display_excl[
                    [
                        "estimator",
                        "real_dgp_rank",
                        "simulator_rank",
                        "rank_diff",
                        "real_dgp_mc_mse",
                        "simulator_mc_mse",
                    ]
                ],
                digits=6,
            )
            lines.append("Excluded own-family estimator comparison")
            lines.append(display_excl.to_string(index=False))
        lines.append("")

    if not summary_rows:
        return ["(no overlapping simulator / REAL_DGP rankings to compare)"]

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["spearman_rho", "mean_abs_rank_diff", "generator"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    summary_df = _format_numeric_df_for_report(summary_df, digits=6)

    out_lines = [
        "SPEARMAN RANK CORRELATION SUMMARY",
        summary_df.to_string(index=False),
        "",
        "ESTIMATOR-LEVEL RANK COMPARISON VS REAL_DGP",
        *lines,
    ]
    if out_lines and out_lines[-1] == "":
        out_lines.pop()
    return out_lines


def _format_distance_summary(dist_df: pd.DataFrame) -> list[str]:
    """
    Distance summaries averaged over MC runs by generator x group.

    Expected raw dist_df columns (from helpers.distribution_distances):
      generator, group, W1, KS, Energy, JS_hist, MeanDiff, StdRatio

    Output:
      one table with mean and sd for each metric by generator and group.
    """
    if dist_df.empty:
        return ["(none)"]

    df = dist_df.copy()

    required = {"generator", "group"}
    if not required.issubset(df.columns):
        return ["(dist_df must contain 'generator' and 'group')"]

    summary_metric_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_sd")]
    if summary_metric_cols:
        agg = df.copy()
    else:
        metric_candidates = ["W1", "KS", "Energy", "JS_hist", "MeanDiff", "StdRatio"]
        metrics = [c for c in metric_candidates if c in df.columns]

        if not metrics:
            return ["(no distance metric columns found)"]

        agg = (
            df.groupby(["generator", "group"], sort=True)[metrics]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Flatten columns
        flat_cols = []
        for col in agg.columns:
            if isinstance(col, tuple):
                left, right = col
                if right == "":
                    flat_cols.append(left)
                else:
                    suffix = "mean" if right == "mean" else "sd"
                    flat_cols.append(f"{left}_{suffix}")
            else:
                flat_cols.append(col)
        agg.columns = flat_cols

    preferred_order = [
        "generator",
        "group",
        "W1_mean", "W1_sd",
        "KS_mean", "KS_sd",
        "Energy_mean", "Energy_sd",
        "JS_hist_mean", "JS_hist_sd",
        "MeanDiff_mean", "MeanDiff_sd",
        "StdRatio_mean", "StdRatio_sd",
    ]
    preferred_order = [c for c in preferred_order if c in agg.columns]

    agg = agg[preferred_order].sort_values(["generator", "group"], kind="stable")
    agg = _format_numeric_df_for_report(agg, digits=6)

    return [agg.to_string(index=False)]


# ==============================================================================
# WRITE REPORT
# ==============================================================================

def write_master_report(
    *,
    out_path: Path,
    perf_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    dgp_description: str = "",
    real_dgp_perf_df: pd.DataFrame | None = None,
    estimator_class_map: dict | None = None,
    project_root: str | None = None,
    execution_summary: dict | None = None,
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 120)
    lines.append("MASTER SUMMARY REPORT — ESTIMATOR PERFORMANCE")
    lines.append("=" * 120)
    lines.append(f"Generated: {now}")
    if project_root:
        lines.append(f"Project root: {project_root}")
    lines.append("")

    if execution_summary:
        lines.append("EXECUTION SUMMARY")
        lines.append("-" * 120)
        for key, value in execution_summary.items():
            if value is None or value == "":
                continue
            label = str(key).replace("_", " ")
            label = label[:1].upper() + label[1:]
            lines.append(f"{label}: {value}")
        lines.append("")

    lines.append("DATA GENERATING PROCESS (DGP)")
    lines.append("-" * 120)
    lines.append(dgp_description if dgp_description else "(not provided)")
    lines.append("")

    if real_dgp_perf_df is not None and not real_dgp_perf_df.empty:
        lines.append("REFERENCE REAL_DGP ESTIMATOR RANKING (lower mc_mse is better)")
        lines.append("-" * 120)
        lines.extend(_format_rankings_by_generator(real_dgp_perf_df))
        lines.append("")

        lines.append("SIMULATOR RANK COMPARISON VS REFERENCE REAL_DGP")
        lines.append("-" * 120)
        lines.extend(_format_rank_comparison_vs_real_dgp(perf_df, real_dgp_perf_df))
        lines.append("")

    lines.append("COMBINED PERFORMANCE TABLE")
    lines.append("-" * 120)
    lines.extend(
        _format_grouped_performance_table(
            perf_df=perf_df,
            estimator_class_map=estimator_class_map,
        )
    )
    lines.append("")

    lines.append("RANKINGS BY GENERATOR (lower mc_mse is better)")
    lines.append("-" * 120)
    lines.extend(_format_rankings_by_generator(perf_df))
    lines.append("")

    lines.append("DISTANCES FOR GENERATED DISTRIBUTION VS REAL DATASET")
    lines.append("AVERAGED OVER MC RUNS, CONDITIONAL ON TREATMENT STATUS AND OVERALL")
    lines.append("-" * 120)
    lines.extend(_format_distance_summary(dist_df))
    lines.append("")

    lines.append("=" * 120)

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ==============================================================================
# PUBLIC ENTRYPOINT
# ==============================================================================
def run_mc_study(
    *,
    generators: dict,
    y_ref: np.ndarray,
    true_ate_by_generator: dict,
    estimators: dict,
    estimator_params: dict,
    X_df: pd.DataFrame,
    A_np: np.ndarray,
    n_mc: int = 500,
    seed_base: int = 100_000,
    alpha: float = 0.05,
    n_workers: int | None = None,
    threads_per_worker: int = 1,
    report_path: str | Path = "MASTER_SUMMARY_REPORT.txt",
    tracked_dir: str | Path | None = None,
    untracked_dir: str | Path | None = None,
    dgp_description: str = "",
    real_dgp_perf_df: pd.DataFrame | None = None,
    estimator_class_map: dict | None = None,
    project_root: str | None = None,
    report_context: dict | None = None,
    save_csv: bool = True,
    write_report: bool = True,
):
    study_t0 = time.perf_counter()
    if n_workers is None:
        n_workers = max(1, available_cpus() - 1)

    all_perf = []
    all_dist = []
    outer_workers = max(1, int(n_workers))
    estimator_threads = max(1, int(threads_per_worker))
    _, process_method = _resolve_mc_process_context()

    for gen_label, gen in generators.items():
        print("\n" + "=" * 80)
        print(
            f"START generator: {gen_label} | "
            f"n_mc={n_mc} | workers={n_workers} | threads/worker={threads_per_worker}"
        )
        print(
            "MC execution mode: chunked worker processes over pre-generated y | "
            f"process_start={process_method} | "
            f"outer_workers={outer_workers} | estimator_threads={estimator_threads}"
        )
        t0 = time.perf_counter()

        true_ate = float(true_ate_by_generator[gen_label])

        est_df, dist_df = run_for_generator(
            gen=gen,
            gen_label=gen_label,
            y_ref=y_ref,
            true_ate=true_ate,
            estimators=estimators,
            estimator_params=estimator_params,
            X_df=X_df,
            A_np=A_np,
            n_mc=n_mc,
            seed_base=seed_base,
            alpha=alpha,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
        )

        dt = time.perf_counter() - t0
        gen_reps_per_sec = (float(n_mc) / dt) if dt > 0 else float("nan")
        gen_secs_per_rep = (dt / float(n_mc)) if int(n_mc) > 0 else float("nan")
        print(
            f"FULL {gen_label} study complete | elapsed: {dt:.2f}s ({dt/60:.1f} min) "
            f"| avg_rate={gen_reps_per_sec:.3f} reps/s | avg_sec/rep={gen_secs_per_rep:.3f}"
        )
        print("=" * 80 + "\n")

        perf_one = summarize_mc(est_df, true_ate)
        perf_one["generator"] = gen_label

        all_perf.append(perf_one)
        all_dist.append(dist_df)

    perf_df = pd.concat(all_perf, ignore_index=True) if all_perf else pd.DataFrame()
    dist_df = pd.concat(all_dist, ignore_index=True) if all_dist else pd.DataFrame()

    # ------------------------------------------------------------------
    # Resolve output directories
    # ------------------------------------------------------------------
    tracked_path = Path(tracked_dir) if tracked_dir is not None else None
    untracked_path = Path(untracked_dir) if untracked_dir is not None else None

    if tracked_path is not None:
        tracked_path.mkdir(parents=True, exist_ok=True)
    if untracked_path is not None:
        untracked_path.mkdir(parents=True, exist_ok=True)

    study_dt = time.perf_counter() - study_t0
    total_replications = int(n_mc) * int(len(generators))
    total_reps_per_sec = (total_replications / study_dt) if study_dt > 0 else float("nan")
    total_secs_per_rep = (study_dt / total_replications) if total_replications > 0 else float("nan")
    print(
        f"TOTAL Monte Carlo study elapsed: {study_dt:.2f}s ({study_dt/60:.1f} min) "
        f"| avg_rate={total_reps_per_sec:.3f} reps/s | avg_sec/rep={total_secs_per_rep:.3f}"
    )

    report_summary = dict(report_context or {})
    report_summary.setdefault("mc_outer_workers", int(outer_workers))
    report_summary.setdefault("threads_per_worker", int(estimator_threads))
    report_summary.setdefault("nominal_worker_threads", int(outer_workers * estimator_threads))
    report_summary.setdefault("process_start_method", str(process_method))
    report_summary.setdefault("runtime_available_cpus", int(available_cpus()))
    report_summary.setdefault("mc_study_elapsed_seconds", f"{study_dt:.2f}")
    report_summary.setdefault("mc_study_elapsed_minutes", f"{study_dt/60:.2f}")
    report_summary.setdefault("mc_total_replications", int(total_replications))
    report_summary.setdefault("mc_avg_reps_per_second", f"{total_reps_per_sec:.3f}")
    report_summary.setdefault("mc_avg_seconds_per_rep", f"{total_secs_per_rep:.3f}")
    submit_epoch = report_summary.get("job_submit_epoch")
    if submit_epoch not in (None, ""):
        try:
            since_submit = max(0.0, time.time() - float(submit_epoch))
            report_summary["elapsed_since_submit_seconds"] = f"{since_submit:.2f}"
            report_summary["elapsed_since_submit_minutes"] = f"{since_submit/60:.2f}"
        except Exception:
            pass

    report_file = None
    if write_report:
        report_path = Path(report_path)
        if tracked_path is not None:
            report_file = tracked_path / report_path.name
        else:
            report_file = report_path

        # ------------------------------------------------------------------
        # Write tracked TXT report
        # ------------------------------------------------------------------
        write_master_report(
            out_path=report_file,
            perf_df=perf_df,
            dist_df=dist_df,
            dgp_description=dgp_description,
            real_dgp_perf_df=real_dgp_perf_df,
            estimator_class_map=estimator_class_map,
            project_root=project_root,
            execution_summary=report_summary,
        )

    # ------------------------------------------------------------------
    # Write untracked CSVs
    # ------------------------------------------------------------------
    if save_csv and untracked_path is not None:
        perf_df.to_csv(untracked_path / "MASTER_mc_summary.csv", index=False)

        if not dist_df.empty and {"generator", "group"}.issubset(dist_df.columns):
            dist_df.to_csv(
                untracked_path / "MASTER_distribution_distances_summary.csv",
                index=False,
            )

    return perf_df, dist_df
