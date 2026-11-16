from __future__ import annotations

# Set thread limits before importing numerical libraries.
import os
import sys
import inspect
from datetime import datetime
from pathlib import Path

# ==============================================================================
def _set_thread_limits(t: int):
    t = int(max(1, t))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            torch_mod.set_num_threads(t)
        except Exception:
            pass

# Keep main process single-threaded by default (workers will override)
_set_thread_limits(1)

# ---- 1) Imports that may initialize BLAS/OpenMP ----
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance, ks_2samp, energy_distance
from scipy.spatial.distance import jensenshannon

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


# ==============================================================================
# CPU detection (affinity + SLURM-aware)
# ==============================================================================
def available_cpus() -> int:
    """
    Best-effort CPU count that respects:
      - Linux cpuset affinity (sched_getaffinity)
      - SLURM_CPUS_PER_TASK, SLURM_CPUS_ON_NODE
    Falls back to os.cpu_count().
    """
    # 1) SLURM (if present)
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(k, None)
        if v is not None:
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                pass

    # 2) affinity (Linux)
    try:
        if hasattr(os, "sched_getaffinity"):
            return int(len(os.sched_getaffinity(0)))
    except Exception:
        pass

    # 3) fallback
    c = os.cpu_count() or 1
    return int(max(1, c))

# ==============================================================================
# DISTANCE HELPERS
# ==============================================================================
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
    A_np  = np.asarray(A_np).ravel()
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

        w1 = float(wasserstein_distance(yr, yg))
        ks = float(ks_2samp(yr, yg).statistic)
        en = float(energy_distance(yr, yg))
        js = float(js_divergence_hist(yr, yg, bins="fd"))

        mean_diff = float(np.mean(yg) - np.mean(yr))
        std_ratio = float(np.std(yg, ddof=1) / (np.std(yr, ddof=1) + 1e-12))

        rows.append({
            "generator": str(label),
            "group": str(grp_name),
            "W1": w1,
            "KS": ks,
            "Energy": en,
            "JS_hist": js,
            "MeanDiff": mean_diff,
            "StdRatio": std_ratio,
        })
    return rows

# ==============================================================================
# kwargs filtering (prevents "unexpected keyword argument" errors)
# ==============================================================================
def _filter_kwargs_for_fn(fn, kwargs: dict) -> dict:
    """
    Keep only kwargs accepted by fn, unless fn has **kwargs.
    """
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        # safest fallback: pass nothing extra
        return {}

# ==============================================================================
# SERIAL runner over a replicate BLOCK (node slicing happens outside)
# ==============================================================================
def run_mc_all_estimators_block(
    *,
    gen_label: str,
    gen_fn,
    seed_base: int,
    y_ref,
    true_ate: float,
    estimators: dict,          # {name: fn}
    estimator_params: dict,     # {name: kwargs}
    r_start: int,
    r_end: int,
    X_df: pd.DataFrame,
    A_np: np.ndarray,
    alpha: float = 0.05,
):
    rows_by_est = {e: [] for e in estimators.keys()}
    dist_rows = []

    y_ref = np.asarray(y_ref).ravel()
    A_np  = np.asarray(A_np).ravel()

    for r in range(int(r_start), int(r_end)):
        seed = int(seed_base + r)
        y = np.asarray(gen_fn(seed)).ravel()

        # generator-level distances (once per replicate)
        dist_rows.extend(distribution_distances(y_gen=y, y_ref=y_ref, A_np=A_np, label=gen_label))

        for est_label, ate_fn in estimators.items():
            kwargs = dict(estimator_params.get(est_label, {}))

            # remove forbidden keys if present
            for bad in ("seed", "random_state", "alpha"):
                kwargs.pop(bad, None)

            kwargs = _filter_kwargs_for_fn(ate_fn, kwargs)

            ate_hat, se_hat, ci_lo, ci_hi, cover = ate_fn(
                X_df, A_np, y,
                true_ate=true_ate,
                seed=seed,
                alpha=alpha,
                **kwargs
            )
            rows_by_est[est_label].append((r, seed, ate_hat, se_hat, ci_lo, ci_hi, cover))

    mc_details_by_est = {
        est: pd.DataFrame(v, columns=["r", "seed", "ate_hat", "se_hat", "ci_lo", "ci_hi", "cover"])
        for est, v in rows_by_est.items()
    }
    dist_df = pd.DataFrame(dist_rows)
    return mc_details_by_est, dist_df

# ==============================================================================
# GENERATOR DIAGNOSTICS HELPERS
#   - reusable for GB / RF cached mean+variance generators
#   - saves TXT/PNG to TRACKED
#   - saves CSVs to UNTRACKED
#   - does NOT interfere with run_mc_study()
# ==============================================================================


# ------------------------------------------------------------------------------
# small utilities
# ------------------------------------------------------------------------------

def _diag_as_1d(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v)
    if a.ndim == 1 and a.shape[0] == n:
        return a.reshape(-1)
    if a.ndim == 2 and a.shape == (n, 1):
        return a[:, 0].reshape(-1)
    if a.ndim == 2 and a.shape == (1, n):
        return a[0, :].reshape(-1)
    raise ValueError(f"{name} has shape {a.shape}; expected (n,), (n,1), or (1,n)")


def _safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: len(x)={x.shape[0]} vs len(y)={y.shape[0]}")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return np.nan
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _sample_from_cached_mu_sigma(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_samples: int = 1,
    seed: int = 123,
    noise: str = "normal",
    t_df: int = 7,
) -> np.ndarray:
    """
    Draws samples from Y = mu + sigma * eps for a fixed treatment arm.
    Returns shape (n,) if n_samples == 1, else (n, S).
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    if mu.shape != sigma.shape:
        raise ValueError(f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")

    rng = np.random.default_rng(int(seed))
    n = mu.shape[0]
    S = int(n_samples)

    if noise == "normal":
        eps = rng.standard_normal((n, S))
    elif noise == "t":
        if int(t_df) <= 2:
            raise ValueError("t_df must be > 2 for finite variance.")
        raw = rng.standard_t(df=int(t_df), size=(n, S))
        eps = raw / np.sqrt(float(t_df) / (float(t_df) - 2.0))
    else:
        raise ValueError("noise must be 'normal' or 't'.")

    out = mu[:, None] + sigma[:, None] * eps
    return out[:, 0] if S == 1 else out


def estimate_cate_ate_from_samples(y0_samps: np.ndarray, y1_samps: np.ndarray):
    """
    y0_samps, y1_samps: shape (n, S)
    """
    y0_samps = np.asarray(y0_samps, dtype=float)
    y1_samps = np.asarray(y1_samps, dtype=float)

    if y0_samps.ndim == 1:
        y0_samps = y0_samps[:, None]
    if y1_samps.ndim == 1:
        y1_samps = y1_samps[:, None]

    if y0_samps.shape != y1_samps.shape:
        raise ValueError(f"Shape mismatch: y0 {y0_samps.shape} vs y1 {y1_samps.shape}")

    d = y1_samps - y0_samps
    cate = d.mean(axis=1)
    cate_sd = d.std(axis=1, ddof=1) if d.shape[1] > 1 else np.full(d.shape[0], np.nan)

    ate_s = d.mean(axis=0)
    ate = float(ate_s.mean())
    ate_se = float(ate_s.std(ddof=1) / np.sqrt(ate_s.size)) if ate_s.size > 1 else np.nan
    return cate, cate_sd, ate, ate_se


def _generator_distance_summary_from_one_draw(
    *,
    y_real: np.ndarray,
    y_synth: np.ndarray,
    A_np: np.ndarray,
    label: str,
) -> pd.DataFrame:
    rows = distribution_distances(
        y_gen=np.asarray(y_synth, dtype=float).reshape(-1),
        y_ref=np.asarray(y_real, dtype=float).reshape(-1),
        A_np=np.asarray(A_np).reshape(-1),
        label=str(label),
    )
    return pd.DataFrame(rows)


def _shared_plot_limits(*arrays, lower_q: float = 0.0, upper_q: float = 1.0) -> tuple[float, float]:
    vals = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return (-1.0, 1.0)
    all_vals = np.concatenate(vals)
    lo = float(np.quantile(all_vals, lower_q))
    hi = float(np.quantile(all_vals, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
    if lo == hi:
        pad = 1.0 if lo == 0.0 else abs(lo) * 0.05
        lo -= pad
        hi += pad
    return lo, hi


def _shared_hist_bins(*arrays, bins: int = 50, lower_q: float = 0.0, upper_q: float = 1.0) -> np.ndarray:
    lo, hi = _shared_plot_limits(*arrays, lower_q=lower_q, upper_q=upper_q)
    return np.linspace(lo, hi, int(bins) + 1)


# ------------------------------------------------------------------------------
# main reusable diagnostics runner
# ------------------------------------------------------------------------------

def run_cached_generator_diagnostics(
    *,
    out_prefix: str,
    generator_label: str,
    X_df: pd.DataFrame,
    A_np,
    Y_np,
    mu0: np.ndarray,
    mu1: np.ndarray,
    sigma0: np.ndarray,
    sigma1: np.ndarray,
    out_tracked: str | Path,
    out_untracked: str | Path,
    noise: str = "normal",
    t_df: int = 7,
    seed_fit: int = 42,
    seed_sample: int = 12345,
    n_samples_mc: int = 10000,
    bivar_cols: list[str] | None = None,
    clip_outcome_quantile_overall: float = 0.995,
    diag_dpi: int = 150,
    outcome_scale: str = "identity",
    write_tracked_artifacts: bool = True,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    """
    Generic diagnostics for cached mean/variance generators.

    Expected inputs
    ---------------
    mu0, mu1, sigma0, sigma1 : arrays of shape (n,)
        Cached potential-outcome means and SDs for A=0 and A=1.
    Y_np : observed/reference outcome on the modeling scale
    A_np : observed treatment assignment

    Saves
    -----
    TRACKED:
      - {out_prefix}_REPORT.txt
      - {out_prefix}_diagnostics.png
      - {out_prefix}_bicovariate_diagnostics.png   (if columns available)

    UNTRACKED:
      - {out_prefix}_correlation_analysis.csv
      - {out_prefix}_individual_results.csv
      - {out_prefix}_summary_statistics.csv
      - {out_prefix}_distribution_distances.csv
    """
    out_tracked = Path(out_tracked)
    out_untracked = Path(out_untracked)
    out_tracked.mkdir(parents=True, exist_ok=True)
    out_untracked.mkdir(parents=True, exist_ok=True)

    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a pandas DataFrame.")

    X_df = X_df.reset_index(drop=True).copy()
    n = len(X_df)
    if outcome_scale not in {"identity", "log"}:
        raise ValueError("outcome_scale must be one of {'identity', 'log'}.")

    mu0 = _diag_as_1d(mu0, n, "mu0")
    mu1 = _diag_as_1d(mu1, n, "mu1")
    sigma0 = _diag_as_1d(sigma0, n, "sigma0")
    sigma1 = _diag_as_1d(sigma1, n, "sigma1")
    A_vec = _diag_as_1d(A_np, n, "A_np").astype(int)
    Y_real_model = _diag_as_1d(Y_np, n, "Y_np")

    if not set(np.unique(A_vec)).issubset({0, 1}):
        raise ValueError("A_np must be binary {0,1}.")

    mask0 = (A_vec == 0)
    mask1 = (A_vec == 1)
    if mask0.sum() == 0 or mask1.sum() == 0:
        raise ValueError(f"One treatment group is empty: n0={mask0.sum()}, n1={mask1.sum()}")

    # ------------------------------------------------------------------
    # draw many potential outcomes to estimate CATE / ATE
    # ------------------------------------------------------------------
    y0 = _sample_from_cached_mu_sigma(
        mu=mu0,
        sigma=sigma0,
        n_samples=int(n_samples_mc),
        seed=int(seed_sample),
        noise=noise,
        t_df=int(t_df),
    )
    y1 = _sample_from_cached_mu_sigma(
        mu=mu1,
        sigma=sigma1,
        n_samples=int(n_samples_mc),
        seed=int(seed_sample) + 1,
        noise=noise,
        t_df=int(t_df),
    )

    cate, cate_sd, ate, ate_se = estimate_cate_ate_from_samples(y0, y1)
    ate_ci = (float(ate - 1.96 * ate_se), float(ate + 1.96 * ate_se)) if np.isfinite(ate_se) else (np.nan, np.nan)

    if outcome_scale == "log":
        ate_pct = (np.exp(ate) - 1.0) * 100.0 if np.isfinite(ate) else np.nan
        ci_lower_pct = (np.exp(ate_ci[0]) - 1.0) * 100.0 if np.isfinite(ate_ci[0]) else np.nan
        ci_upper_pct = (np.exp(ate_ci[1]) - 1.0) * 100.0 if np.isfinite(ate_ci[1]) else np.nan
    else:
        ate_pct = np.nan
        ci_lower_pct = np.nan
        ci_upper_pct = np.nan

    y0_mean = np.asarray(y0, dtype=float).mean(axis=1)
    y1_mean = np.asarray(y1, dtype=float).mean(axis=1)

    # ------------------------------------------------------------------
    # one synthetic realized dataset using observed A
    # ------------------------------------------------------------------
    y0_1 = _sample_from_cached_mu_sigma(
        mu=mu0, sigma=sigma0, n_samples=1,
        seed=int(seed_sample) + 10, noise=noise, t_df=int(t_df),
    )
    y1_1 = _sample_from_cached_mu_sigma(
        mu=mu1, sigma=sigma1, n_samples=1,
        seed=int(seed_sample) + 11, noise=noise, t_df=int(t_df),
    )

    Y_synth_model = np.where(A_vec == 1, y1_1, y0_1).reshape(-1)

    if outcome_scale == "log":
        Y_real_outcome = np.exp(Y_real_model)
        Y_synth_outcome = np.exp(Y_synth_model)
        y0_mean_outcome = np.exp(y0_mean)
        y1_mean_outcome = np.exp(y1_mean)
        effect_label = "log scale"
        outcome_label = "Outcome"
        observed_level_label = "Observed mean log Y"
        cf_level_label = "Counterfactual mean log Y"
    else:
        Y_real_outcome = Y_real_model.copy()
        Y_synth_outcome = Y_synth_model.copy()
        y0_mean_outcome = y0_mean.copy()
        y1_mean_outcome = y1_mean.copy()
        effect_label = "outcome scale"
        outcome_label = "Outcome"
        observed_level_label = "Observed mean outcome"
        cf_level_label = "Counterfactual mean outcome"

    # ------------------------------------------------------------------
    # counterfactual analysis by observed A
    # ------------------------------------------------------------------
    Y0_obs = Y_real_outcome[mask0]
    Y1_obs = Y_real_outcome[mask1]
    Y0_cf_if1 = y1_mean[mask0]
    Y1_cf_if0 = y0_mean[mask1]

    # ------------------------------------------------------------------
    # correlations
    # ------------------------------------------------------------------
    corr_rows = []
    for col in X_df.columns:
        xcol = X_df[col].to_numpy()
        real_corr = _safe_corr(xcol, Y_real_outcome)
        syn_corr = _safe_corr(xcol, Y_synth_outcome)
        diff = syn_corr - real_corr if np.isfinite(real_corr) and np.isfinite(syn_corr) else np.nan
        absdiff = abs(diff) if np.isfinite(diff) else np.nan
        corr_rows.append((col, real_corr, syn_corr, diff, absdiff))

    real_corr_A = _safe_corr(A_vec, Y_real_outcome)
    syn_corr_A = _safe_corr(A_vec, Y_synth_outcome)
    diffA = syn_corr_A - real_corr_A if np.isfinite(real_corr_A) and np.isfinite(syn_corr_A) else np.nan
    absdiffA = abs(diffA) if np.isfinite(diffA) else np.nan
    corr_rows.append(("Treatment (A)", real_corr_A, syn_corr_A, diffA, absdiffA))

    correlation_df = pd.DataFrame(
        corr_rows,
        columns=["variable", "real_corr", "synthetic_corr", "difference", "abs_difference"]
    )

    mean_abs_diff = float(np.nanmean(correlation_df["abs_difference"].to_numpy()))
    max_abs_diff = float(np.nanmax(correlation_df["abs_difference"].to_numpy()))

    # ------------------------------------------------------------------
    # distances for one synthetic draw vs real
    # ------------------------------------------------------------------
    dist_df = _generator_distance_summary_from_one_draw(
        y_real=Y_real_outcome,
        y_synth=Y_synth_outcome,
        A_np=A_vec,
        label=generator_label,
    )

    # ------------------------------------------------------------------
    # save CSVs
    # ------------------------------------------------------------------
    corr_csv = out_untracked / f"{out_prefix}_correlation_analysis.csv"
    indiv_csv = out_untracked / f"{out_prefix}_individual_results.csv"
    summ_csv = out_untracked / f"{out_prefix}_summary_statistics.csv"
    dist_csv = out_untracked / f"{out_prefix}_distribution_distances.csv"

    cate_1d = _diag_as_1d(cate, n, "cate")
    cate_sd_1d = _diag_as_1d(cate_sd, n, "cate_sd")
    y0_mean_1d = _diag_as_1d(y0_mean, n, "y0_mean")
    y1_mean_1d = _diag_as_1d(y1_mean, n, "y1_mean")
    Y_synth_model_1d = _diag_as_1d(Y_synth_model, n, "Y_synth_model")
    Y_synth_outcome_1d = _diag_as_1d(Y_synth_outcome, n, "Y_synth_outcome")
    Y_model_1d = _diag_as_1d(Y_real_model, n, "Y_real_model")
    Y_outcome_1d = _diag_as_1d(Y_real_outcome, n, "Y_real_outcome")

    results_df = pd.DataFrame({
        "cate": cate_1d,
        "cate_sd_mc": cate_sd_1d,
        "mu0": mu0,
        "mu1": mu1,
        "sigma0": sigma0,
        "sigma1": sigma1,
        "y0_mean_model_scale": y0_mean_1d,
        "y1_mean_model_scale": y1_mean_1d,
        "y0_mean_outcome": y0_mean_outcome,
        "y1_mean_outcome": y1_mean_outcome,
        "y_synth_model_scale": Y_synth_model_1d,
        "y_synth_outcome": Y_synth_outcome_1d,
        "observed_A": A_vec,
        "observed_model_scale": Y_model_1d,
        "observed_outcome": Y_outcome_1d,
    })
    # Backward-compatible aliases for older analyses that expected *_log names.
    results_df["y0_mean_log"] = y0_mean_1d
    results_df["y1_mean_log"] = y1_mean_1d
    results_df["y_synth_log"] = Y_synth_model_1d
    results_df["observed_logY"] = Y_model_1d
    results_df = pd.concat([results_df, X_df.reset_index(drop=True)], axis=1)

    summary_df = pd.DataFrame([{
        "generator_label": str(generator_label),
        "seed_fit": int(seed_fit),
        "seed_sample": int(seed_sample),
        "n_samples_mc": int(n_samples_mc),
        "noise": str(noise),
        "t_df": int(t_df),
        "n_obs": int(n),
        "n_control": int(mask0.sum()),
        "n_treated": int(mask1.sum()),
        "ate": float(ate),
        "ate_se": float(ate_se) if np.isfinite(ate_se) else np.nan,
        "ate_ci_lower": float(ate_ci[0]) if np.isfinite(ate_ci[0]) else np.nan,
        "ate_ci_upper": float(ate_ci[1]) if np.isfinite(ate_ci[1]) else np.nan,
        "ate_pct": float(ate_pct) if np.isfinite(ate_pct) else np.nan,
        "ate_pct_ci_lower": float(ci_lower_pct) if np.isfinite(ci_lower_pct) else np.nan,
        "ate_pct_ci_upper": float(ci_upper_pct) if np.isfinite(ci_upper_pct) else np.nan,
        "mean_cate": float(np.mean(cate_1d)),
        "std_cate": float(np.std(cate_1d, ddof=1)),
        "min_cate": float(np.min(cate_1d)),
        "max_cate": float(np.max(cate_1d)),
        "mean_abs_corr_diff": float(mean_abs_diff),
        "max_abs_corr_diff": float(max_abs_diff),
    }])

    if not write_tracked_artifacts:
        return {
            "correlation_df": correlation_df,
            "results_df": results_df,
            "summary_df": summary_df,
            "dist_df": dist_df,
            "ate": float(ate),
            "ate_se": float(ate_se) if np.isfinite(ate_se) else np.nan,
            "ate_ci": ate_ci,
            "ate_pct": float(ate_pct) if np.isfinite(ate_pct) else np.nan,
            "report_txt": None,
            "diag_png": None,
            "bivar_png": None,
            "corr_csv": None,
            "indiv_csv": None,
            "summ_csv": None,
            "dist_csv": None,
        }

    correlation_df.to_csv(corr_csv, index=False)
    results_df.to_csv(indiv_csv, index=False)
    summary_df.to_csv(summ_csv, index=False)
    dist_df.to_csv(dist_csv, index=False)

    # ------------------------------------------------------------------
    # diagnostics figure
    # ------------------------------------------------------------------
    diag_png = out_tracked / f"{out_prefix}_diagnostics.png"

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(cate_1d, bins=50, alpha=0.7, edgecolor="black")
    ax1.axvline(cate_1d.mean(), linestyle="--", linewidth=2, label=f"Mean: {cate_1d.mean():.4f}")
    ax1.set_title(f"CATE Distribution ({generator_label}, {effect_label})")
    ax1.set_xlabel("CATE")
    ax1.set_ylabel("Count")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    q = np.linspace(0.01, float(clip_outcome_quantile_overall), 400)
    real_q = np.quantile(Y_real_outcome, q)
    syn_q = np.quantile(Y_synth_outcome, q)
    ax2.scatter(real_q, syn_q, alpha=0.5, s=15)
    qmin, qmax = _shared_plot_limits(real_q, syn_q)
    ax2.plot([qmin, qmax], [qmin, qmax], "r--", alpha=0.6)
    ax2.set_xlim(qmin, qmax)
    ax2.set_ylim(qmin, qmax)
    ax2.set_title("Q-Q Plot: Real vs Synthetic Outcome")
    ax2.set_xlabel("Real quantiles")
    ax2.set_ylabel("Synthetic quantiles")
    ax2.grid(alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(Y_real_outcome, Y_synth_outcome, alpha=0.25, s=10)
    scatter_min, scatter_max = _shared_plot_limits(
        Y_real_outcome,
        Y_synth_outcome,
        lower_q=1.0 - float(clip_outcome_quantile_overall),
        upper_q=float(clip_outcome_quantile_overall),
    )
    ax3.plot([scatter_min, scatter_max], [scatter_min, scatter_max], "r--", alpha=0.6)
    ax3.set_xlim(scatter_min, scatter_max)
    ax3.set_ylim(scatter_min, scatter_max)
    ax3.set_title("Real vs Synthetic Outcomes")
    ax3.set_xlabel("Real outcome")
    ax3.set_ylabel("Synthetic outcome")
    ax3.grid(alpha=0.3)
    ax3.set_aspect("equal", adjustable="box")

    ax4 = fig.add_subplot(gs[1, 0])
    overall_lo, overall_hi = _shared_plot_limits(
        Y_real_outcome,
        Y_synth_outcome,
        lower_q=1.0 - float(clip_outcome_quantile_overall),
        upper_q=float(clip_outcome_quantile_overall),
    )
    real_tr = Y_real_outcome[(Y_real_outcome >= overall_lo) & (Y_real_outcome <= overall_hi)]
    syn_tr = Y_synth_outcome[(Y_synth_outcome >= overall_lo) & (Y_synth_outcome <= overall_hi)]
    overall_bins = _shared_hist_bins(real_tr, syn_tr, bins=50)
    ax4.hist(real_tr, bins=overall_bins, density=True, alpha=0.5, label="Real", edgecolor="black")
    ax4.hist(syn_tr, bins=overall_bins, density=True, alpha=0.5, label="Synthetic", edgecolor="black")
    ax4.set_title(f"Overall Distribution (≤ {100*clip_outcome_quantile_overall:.1f}% quantile)")
    ax4.set_title(f"Overall Distribution (shared {100*clip_outcome_quantile_overall:.1f}% window)")
    ax4.set_xlabel(outcome_label)
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    control_lo, control_hi = _shared_plot_limits(
        Y_real_outcome[mask0],
        Y_synth_outcome[mask0],
        lower_q=1.0 - float(clip_outcome_quantile_overall),
        upper_q=float(clip_outcome_quantile_overall),
    )
    real_control = Y_real_outcome[mask0]
    syn_control = Y_synth_outcome[mask0]
    real_control = real_control[(real_control >= control_lo) & (real_control <= control_hi)]
    syn_control = syn_control[(syn_control >= control_lo) & (syn_control <= control_hi)]
    control_bins = _shared_hist_bins(real_control, syn_control, bins=35)
    ax5.hist(real_control, bins=control_bins, density=True, alpha=0.5, label="Real", edgecolor="black")
    ax5.hist(syn_control, bins=control_bins, density=True, alpha=0.5, label="Synthetic", edgecolor="black")
    ax5.set_title("Control Group (A=0) Distribution")
    ax5.set_xlabel(outcome_label)
    ax5.set_ylabel("Density")
    ax5.legend()
    ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    treated_lo, treated_hi = _shared_plot_limits(
        Y_real_outcome[mask1],
        Y_synth_outcome[mask1],
        lower_q=1.0 - float(clip_outcome_quantile_overall),
        upper_q=float(clip_outcome_quantile_overall),
    )
    real_treated = Y_real_outcome[mask1]
    syn_treated = Y_synth_outcome[mask1]
    real_treated = real_treated[(real_treated >= treated_lo) & (real_treated <= treated_hi)]
    syn_treated = syn_treated[(syn_treated >= treated_lo) & (syn_treated <= treated_hi)]
    treated_bins = _shared_hist_bins(real_treated, syn_treated, bins=35)
    ax6.hist(real_treated, bins=treated_bins, density=True, alpha=0.5, label="Real", edgecolor="black")
    ax6.hist(syn_treated, bins=treated_bins, density=True, alpha=0.5, label="Synthetic", edgecolor="black")
    ax6.set_title("Treated Group (A=1) Distribution")
    ax6.set_xlabel(outcome_label)
    ax6.set_ylabel("Density")
    ax6.legend()
    ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, :])
    top15 = correlation_df.nlargest(15, "abs_difference").iloc[::-1]
    ax7.barh(top15["variable"].astype(str), top15["difference"], alpha=0.8)
    ax7.axvline(0, linestyle="--", alpha=0.6)
    ax7.set_title("Top 15 Correlation Differences (Synthetic - Real)")
    ax7.set_xlabel("Correlation difference")
    ax7.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(diag_png, bbox_inches="tight", dpi=int(diag_dpi))
    plt.close(fig)

    # ------------------------------------------------------------------
    # optional bivariate figure
    # ------------------------------------------------------------------
    bivar_png = None
    if bivar_cols is None:
        bivar_cols = ["age", "diploma", "compas"]

    missing = [c for c in bivar_cols if c not in X_df.columns]
    if len(missing) == 0 and len(bivar_cols) >= 3:
        bivar_png = out_tracked / f"{out_prefix}_bicovariate_diagnostics.png"

        x1_name, x2_name, x3_name = bivar_cols[:3]
        x1 = X_df[x1_name].to_numpy()
        x2 = X_df[x2_name].to_numpy()
        x3 = X_df[x3_name].to_numpy()

        vmin, vmax = _shared_plot_limits(
            Y_real_outcome,
            Y_synth_outcome,
            lower_q=1.0 - float(clip_outcome_quantile_overall),
            upper_q=float(clip_outcome_quantile_overall),
        )

        money_fmt = FuncFormatter(lambda x, pos: f"${x/1:.0f}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=int(diag_dpi))
        fig.suptitle(f"Bi-Covariate Diagnostics: Real vs {generator_label}-Synthetic", fontsize=18, y=0.98)

        pairs = [
            (x1_name, x1, x2_name, x2),
            (x1_name, x1, x3_name, x3),
            (x2_name, x2, x3_name, x3),
        ]

        def scatter_panel(ax, x, y, c, title):
            sc = ax.scatter(x, y, c=c, s=18, alpha=0.7, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(title.split(": ")[1].split(" vs ")[0])
            ax.set_ylabel(title.split(" vs ")[1].split(" (")[0])
            ax.grid(alpha=0.25)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Outcome")
            cb.ax.yaxis.set_major_formatter(money_fmt)
            return sc

        for j, (xname, x, yname, y) in enumerate(pairs):
            scatter_panel(axes[0, j], x, y, Y_real_outcome, f"Real: {xname} vs {yname} (colored by Outcome)")

        for j, (xname, x, yname, y) in enumerate(pairs):
            scatter_panel(axes[1, j], x, y, Y_synth_outcome, f"{generator_label}-Synthetic: {xname} vs {yname} (colored by Outcome)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(bivar_png, bbox_inches="tight", dpi=int(diag_dpi))
        plt.close(fig)

    # ------------------------------------------------------------------
    # text report
    # ------------------------------------------------------------------
    report_txt = out_tracked / f"{out_prefix}_REPORT.txt"

    report = f"""
{'='*80}
{generator_label} GENERATOR DIAGNOSTICS REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SETTINGS
- out_prefix: {out_prefix}
- generator_label: {generator_label}
- n_samples per unit: {n_samples_mc}
- noise: {noise} (t_df={t_df})
- seed_fit: {seed_fit}
- seed_sample: {seed_sample}

DATA
- N: {len(X_df)}
- Control (A=0): {int(mask0.sum())}
- Treated (A=1): {int(mask1.sum())}

ATE
- ATE (log): {ate:.6f} ± {ate_se:.6f}
- 95% CI (log): [{ate_ci[0]:.6f}, {ate_ci[1]:.6f}]
- ATE (%): {ate_pct:.3f}%
- 95% CI (%): [{ci_lower_pct:.3f}%, {ci_upper_pct:.3f}%]

CATE (log)
- Mean: {cate_1d.mean():.6f}
- Std:  {cate_1d.std(ddof=1):.6f}
- Min:  {cate_1d.min():.6f}
- Max:  {cate_1d.max():.6f}

COUNTERFACTUAL ANALYSIS BY OBSERVED A
A=0:
- Observed mean log Y: {Y0_obs.mean():.6f}
- Counterfactual if A=1: {Y0_cf_if1.mean():.6f}
- Mean individual effect: {(Y0_cf_if1 - Y0_obs).mean():.6f}
- Wage gap (%): {(np.exp(Y0_cf_if1).mean() / np.exp(Y0_obs).mean() - 1.0) * 100.0:.3f}%

A=1:
- Observed mean log Y: {Y1_obs.mean():.6f}
- Counterfactual if A=0: {Y1_cf_if0.mean():.6f}
- Mean individual effect: {(Y1_obs - Y1_cf_if0).mean():.6f}
- Wage gap (%): {(np.exp(Y1_obs).mean() / np.exp(Y1_cf_if0).mean() - 1.0) * 100.0:.3f}%

CORRELATION DIAGNOSTICS (outcome scale)
- Mean abs corr diff: {mean_abs_diff:.6f}
- Max  abs corr diff: {max_abs_diff:.6f}

FILES GENERATED
- {corr_csv}
- {indiv_csv}
- {summ_csv}
- {dist_csv}
- {diag_png}
- {bivar_png if bivar_png is not None else "(bivariate figure skipped)"}
"""
    if outcome_scale == "log":
        ate_extra_lines = (
            f"- ATE (%): {ate_pct:.3f}%\n"
            f"- 95% CI (%): [{ci_lower_pct:.3f}%, {ci_upper_pct:.3f}%]"
        )
        cf_extra_a0 = f"- Wage gap (%): {(np.exp(Y0_cf_if1).mean() / np.exp(Y0_obs).mean() - 1.0) * 100.0:.3f}%"
        cf_extra_a1 = f"- Wage gap (%): {(np.exp(Y1_obs).mean() / np.exp(Y1_cf_if0).mean() - 1.0) * 100.0:.3f}%"
    else:
        ate_extra_lines = "- ATE (%): n/a on identity outcome scale\n- 95% CI (%): n/a on identity outcome scale"
        cf_extra_a0 = "- Wage gap (%): n/a on identity outcome scale"
        cf_extra_a1 = "- Wage gap (%): n/a on identity outcome scale"

    report = f"""
{'='*80}
{generator_label} GENERATOR DIAGNOSTICS REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SETTINGS
- out_prefix: {out_prefix}
- generator_label: {generator_label}
- n_samples per unit: {n_samples_mc}
- noise: {noise} (t_df={t_df})
- outcome_scale: {outcome_scale}
- seed_fit: {seed_fit}
- seed_sample: {seed_sample}

DATA
- N: {len(X_df)}
- Control (A=0): {int(mask0.sum())}
- Treated (A=1): {int(mask1.sum())}

ATE
- ATE ({effect_label}): {ate:.6f} +/- {ate_se:.6f}
- 95% CI ({effect_label}): [{ate_ci[0]:.6f}, {ate_ci[1]:.6f}]
{ate_extra_lines}

CATE ({effect_label})
- Mean: {cate_1d.mean():.6f}
- Std:  {cate_1d.std(ddof=1):.6f}
- Min:  {cate_1d.min():.6f}
- Max:  {cate_1d.max():.6f}

COUNTERFACTUAL ANALYSIS BY OBSERVED A
A=0:
- {observed_level_label}: {Y0_obs.mean():.6f}
- {cf_level_label} if A=1: {Y0_cf_if1.mean():.6f}
- Mean individual effect: {(Y0_cf_if1 - Y0_obs).mean():.6f}
{cf_extra_a0}

A=1:
- {observed_level_label}: {Y1_obs.mean():.6f}
- {cf_level_label} if A=0: {Y1_cf_if0.mean():.6f}
- Mean individual effect: {(Y1_obs - Y1_cf_if0).mean():.6f}
{cf_extra_a1}

CORRELATION DIAGNOSTICS (outcome scale)
- Mean abs corr diff: {mean_abs_diff:.6f}
- Max  abs corr diff: {max_abs_diff:.6f}

FILES GENERATED
- {corr_csv}
- {indiv_csv}
- {summ_csv}
- {dist_csv}
- {diag_png}
- {bivar_png if bivar_png is not None else "(bivariate figure skipped)"}
"""
    report_txt.write_text(report, encoding="utf-8")

    return {
        "correlation_df": correlation_df,
        "results_df": results_df,
        "summary_df": summary_df,
        "dist_df": dist_df,
        "ate": float(ate),
        "ate_se": float(ate_se) if np.isfinite(ate_se) else np.nan,
        "ate_ci": ate_ci,
        "ate_pct": float(ate_pct) if np.isfinite(ate_pct) else np.nan,
        "report_txt": report_txt,
        "diag_png": diag_png,
        "bivar_png": bivar_png,
        "corr_csv": corr_csv,
        "indiv_csv": indiv_csv,
        "summ_csv": summ_csv,
        "dist_csv": dist_csv,
    }

def run_wgan_generator_diagnostics(
    *,
    out_prefix: str,
    generator_label: str,
    fitted_wgan,
    X_df,
    A_np,
    Y_np,
    out_tracked,
    out_untracked,
    seed_fit: int = 42,
    seed_sample: int = 12345,
    n_samples_mc: int = 10000,
    bivar_cols=None,
    write_tracked_artifacts: bool = True,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    if bivar_cols is None:
        bivar_cols = []

    out_tracked = Path(out_tracked)
    out_untracked = Path(out_untracked)
    out_tracked.mkdir(parents=True, exist_ok=True)
    out_untracked.mkdir(parents=True, exist_ok=True)

    X_np = X_df.to_numpy(dtype=np.float32)
    A_np = np.asarray(A_np).reshape(-1).astype(int)
    Y_np = np.asarray(Y_np, dtype=float).reshape(-1)

    # Structural Monte Carlo draws under both potential outcomes
    y0_draws = fitted_wgan.sample_y0(X_np, n_samples=n_samples_mc, seed=seed_sample)
    y1_draws = fitted_wgan.sample_y1(X_np, n_samples=n_samples_mc, seed=seed_sample + 1)

    # Ensure 2D
    if y0_draws.ndim == 1:
        y0_draws = y0_draws[:, None]
    if y1_draws.ndim == 1:
        y1_draws = y1_draws[:, None]

    mu0_hat = y0_draws.mean(axis=1)
    mu1_hat = y1_draws.mean(axis=1)
    sigma0_hat = y0_draws.std(axis=1, ddof=1)
    sigma1_hat = y1_draws.std(axis=1, ddof=1)

    # Observed synthetic sample
    y_obs_syn = np.where(A_np == 1, y1_draws[:, 0], y0_draws[:, 0])

    # Summary table
    diag_df = pd.DataFrame({
        "mu0_hat": mu0_hat,
        "mu1_hat": mu1_hat,
        "sigma0_hat": sigma0_hat,
        "sigma1_hat": sigma1_hat,
        "cate_hat": mu1_hat - mu0_hat,
        "A": A_np,
        "Y_real": Y_np,
        "Y_syn": y_obs_syn,
    }, index=X_df.index)

    rowlevel_path = out_untracked / f"{out_prefix}_diagnostics_rowlevel.csv"

    # Scalar summaries
    checkpoint_log_control = list(getattr(fitted_wgan, "checkpoint_log_control", []) or [])
    checkpoint_log_treated = list(getattr(fitted_wgan, "checkpoint_log_treated", []) or [])
    best_epoch_control = getattr(fitted_wgan, "best_epoch_control", None)
    best_epoch_treated = getattr(fitted_wgan, "best_epoch_treated", None)
    best_w1_scaled_control = getattr(fitted_wgan, "best_w1_scaled_control", None)
    best_w1_scaled_treated = getattr(fitted_wgan, "best_w1_scaled_treated", None)

    summary = {
        "generator_label": generator_label,
        "seed_fit": int(seed_fit),
        "seed_sample": int(seed_sample),
        "n_obs": int(len(X_df)),
        "n_samples_mc": int(n_samples_mc),
        "mean_Y_real": float(np.mean(Y_np)),
        "mean_Y_syn": float(np.mean(y_obs_syn)),
        "sd_Y_real": float(np.std(Y_np, ddof=1)),
        "sd_Y_syn": float(np.std(y_obs_syn, ddof=1)),
        "mean_mu0": float(np.mean(mu0_hat)),
        "mean_mu1": float(np.mean(mu1_hat)),
        "mean_sigma0": float(np.mean(sigma0_hat)),
        "mean_sigma1": float(np.mean(sigma1_hat)),
        "true_ate_mc": float(np.mean(mu1_hat - mu0_hat)),
        "best_epoch_control": None if best_epoch_control is None else int(best_epoch_control),
        "best_epoch_treated": None if best_epoch_treated is None else int(best_epoch_treated),
        "best_w1_scaled_control": (
            np.nan if best_w1_scaled_control is None else float(best_w1_scaled_control)
        ),
        "best_w1_scaled_treated": (
            np.nan if best_w1_scaled_treated is None else float(best_w1_scaled_treated)
        ),
    }

    summary_path = out_tracked / f"{out_prefix}_diagnostics_summary.txt"

    checkpoint_rows = []
    for arm_label, log in (("control", checkpoint_log_control), ("treated", checkpoint_log_treated)):
        for row in log:
            checkpoint_rows.append(
                {
                    "arm": arm_label,
                    "epoch": int(row.get("epoch")),
                    "mean_w1_scaled": float(row.get("mean_w1_scaled")),
                }
            )

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    checkpoint_csv_path = out_untracked / f"{out_prefix}_checkpoint_wdist.csv"

    if not write_tracked_artifacts:
        return {
            "summary": summary,
            "rowlevel_path": None,
            "summary_path": None,
            "report_path": None,
            "checkpoint_csv_path": None,
            "checkpoint_df": checkpoint_df,
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    diag_df.to_csv(rowlevel_path, index=True)
    if not checkpoint_df.empty:
        checkpoint_df.to_csv(checkpoint_csv_path, index=False)

    best_w1_control_str = (
        f"{float(best_w1_scaled_control):.6f}"
        if best_w1_scaled_control is not None else "nan"
    )
    best_w1_treated_str = (
        f"{float(best_w1_scaled_treated):.6f}"
        if best_w1_scaled_treated is not None else "nan"
    )

    control_history = list(getattr(getattr(fitted_wgan, "wgan_control", None), "training_history", []) or [])
    treated_history = list(getattr(getattr(fitted_wgan, "wgan_treated", None), "training_history", []) or [])

    diag_png = out_tracked / f"{out_prefix}_diagnostics.png"
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if control_history:
        ax1.plot(np.arange(1, len(control_history) + 1), [h.get("wasserstein_distance", np.nan) for h in control_history], label="Control")
    if treated_history:
        ax1.plot(np.arange(1, len(treated_history) + 1), [h.get("wasserstein_distance", np.nan) for h in treated_history], label="Treated")
    ax1.set_title("Training W-Distance")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("W-distance proxy")
    ax1.grid(alpha=0.3)
    if control_history or treated_history:
        ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    if control_history:
        ax2.plot(np.arange(1, len(control_history) + 1), [h.get("g_loss", np.nan) for h in control_history], label="Control G")
        ax2.plot(np.arange(1, len(control_history) + 1), [h.get("c_loss", np.nan) for h in control_history], label="Control C")
    if treated_history:
        ax2.plot(np.arange(1, len(treated_history) + 1), [h.get("g_loss", np.nan) for h in treated_history], label="Treated G", linestyle="--")
        ax2.plot(np.arange(1, len(treated_history) + 1), [h.get("c_loss", np.nan) for h in treated_history], label="Treated C", linestyle="--")
    ax2.set_title("Training Losses")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    if control_history or treated_history:
        ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(Y_np, bins=40, alpha=0.6, density=True, label="Observed")
    ax3.hist(y_obs_syn, bins=40, alpha=0.6, density=True, label=f"{generator_label} synthetic")
    ax3.set_title("Overall Outcome Distribution")
    ax3.set_xlabel("Outcome")
    ax3.set_ylabel("Density")
    ax3.grid(alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(Y_np[A_np == 0], bins=35, alpha=0.6, density=True, label="Observed")
    ax4.hist(y_obs_syn[A_np == 0], bins=35, alpha=0.6, density=True, label="Synthetic")
    ax4.set_title("Control Group Distribution")
    ax4.set_xlabel("Outcome")
    ax4.set_ylabel("Density")
    ax4.grid(alpha=0.3)
    ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(Y_np[A_np == 1], bins=35, alpha=0.6, density=True, label="Observed")
    ax5.hist(y_obs_syn[A_np == 1], bins=35, alpha=0.6, density=True, label="Synthetic")
    ax5.set_title("Treated Group Distribution")
    ax5.set_xlabel("Outcome")
    ax5.set_ylabel("Density")
    ax5.grid(alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(Y_np, y_obs_syn, alpha=0.25, s=10)
    mn = float(min(np.min(Y_np), np.min(y_obs_syn)))
    mx = float(max(np.max(Y_np), np.max(y_obs_syn)))
    ax6.plot([mn, mx], [mn, mx], "r--", alpha=0.6)
    ax6.set_title("Observed vs Synthetic Outcomes")
    ax6.set_xlabel("Observed outcome")
    ax6.set_ylabel("Synthetic outcome")
    ax6.grid(alpha=0.3)
    ax6.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(diag_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    bivar_png = None
    missing = [c for c in bivar_cols if c not in X_df.columns]
    if len(missing) == 0 and len(bivar_cols) >= 3:
        bivar_png = out_tracked / f"{out_prefix}_bicovariate_diagnostics.png"

        x1_name, x2_name, x3_name = bivar_cols[:3]
        x1 = X_df[x1_name].to_numpy()
        x2 = X_df[x2_name].to_numpy()
        x3 = X_df[x3_name].to_numpy()

        vmin = float(np.nanmin([Y_np.min(), y_obs_syn.min()]))
        vmax = float(np.nanmax([Y_np.max(), y_obs_syn.max()]))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)
        fig.suptitle(f"Bi-Covariate Diagnostics: Real vs {generator_label}-Synthetic", fontsize=18, y=0.98)

        pairs = [
            (x1_name, x1, x2_name, x2),
            (x1_name, x1, x3_name, x3),
            (x2_name, x2, x3_name, x3),
        ]

        def scatter_panel(ax, x, y, c, title):
            sc = ax.scatter(x, y, c=c, s=18, alpha=0.7, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(title.split(": ")[1].split(" vs ")[0])
            ax.set_ylabel(title.split(" vs ")[1].split(" (")[0])
            ax.grid(alpha=0.25)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Outcome")
            return sc

        for j, (xname, x, yname, y) in enumerate(pairs):
            scatter_panel(axes[0, j], x, y, Y_np, f"Real: {xname} vs {yname} (colored by Outcome)")

        for j, (xname, x, yname, y) in enumerate(pairs):
            scatter_panel(axes[1, j], x, y, y_obs_syn, f"{generator_label}-Synthetic: {xname} vs {yname} (colored by Outcome)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(bivar_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

    report_path = out_tracked / f"{out_prefix}_REPORT.txt"
    checkpoint_lines = []
    if checkpoint_df.empty:
        checkpoint_lines.append("(no checkpoint log available on fitted_wgan)")
    else:
        checkpoint_lines.append("CONTROL CHECKPOINTS")
        checkpoint_lines.append("-" * 80)
        checkpoint_lines.append(
            checkpoint_df.loc[checkpoint_df["arm"] == "control", ["epoch", "mean_w1_scaled"]]
            .to_string(index=False)
        )
        checkpoint_lines.append("")
        checkpoint_lines.append("TREATED CHECKPOINTS")
        checkpoint_lines.append("-" * 80)
        checkpoint_lines.append(
            checkpoint_df.loc[checkpoint_df["arm"] == "treated", ["epoch", "mean_w1_scaled"]]
            .to_string(index=False)
        )

    report = f"""
{'='*80}
WGAN TRAINING REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SETTINGS
- out_prefix: {out_prefix}
- generator_label: {generator_label}
- n_obs: {len(X_df)}
- n_samples_mc: {n_samples_mc}
- seed_fit: {seed_fit}
- seed_sample: {seed_sample}

BEST CHECKPOINTS
- Control best epoch: {best_epoch_control}
- Control best mean exact W1 (scaled): {best_w1_control_str}
- Treated best epoch: {best_epoch_treated}
- Treated best mean exact W1 (scaled): {best_w1_treated_str}

OBSERVED VS SYNTHETIC OUTCOMES
- Mean Y_real: {float(np.mean(Y_np)):.6f}
- Mean Y_syn:  {float(np.mean(y_obs_syn)):.6f}
- SD Y_real:   {float(np.std(Y_np, ddof=1)):.6f}
- SD Y_syn:    {float(np.std(y_obs_syn, ddof=1)):.6f}
- Mean mu0:    {float(np.mean(mu0_hat)):.6f}
- Mean mu1:    {float(np.mean(mu1_hat)):.6f}
- Mean sigma0: {float(np.mean(sigma0_hat)):.6f}
- Mean sigma1: {float(np.mean(sigma1_hat)):.6f}
- MC ATE:      {float(np.mean(mu1_hat - mu0_hat)):.6f}

CHECKPOINT EVOLUTION OF W-DISTANCE
{'-'*80}
{chr(10).join(checkpoint_lines)}

FILES GENERATED
- {summary_path}
- {report_path}
- {out_untracked / f"{out_prefix}_diagnostics_rowlevel.csv"}
- {checkpoint_csv_path if not checkpoint_df.empty else "(checkpoint csv skipped)"}
- {diag_png}
- {bivar_png if bivar_png is not None else "(bicovariate figure skipped)"}
"""
    report_path.write_text(report.strip() + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "rowlevel_path": str(out_untracked / f"{out_prefix}_diagnostics_rowlevel.csv"),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "checkpoint_csv_path": str(checkpoint_csv_path) if not checkpoint_df.empty else None,
        "checkpoint_df": checkpoint_df,
    }



def fit_gb_mean_and_variance_cv(
    X_df: pd.DataFrame,
    A_df,
    Y_df,
    *,
    seed: int = 42,
    n_splits: int = 5,
    # mean model hyperparams (MUST match what you want the generator to represent)
    max_iter_mean: int = 500,
    learning_rate_mean: float = 0.05,
    max_depth_mean: int | None = 6,
    # variance model hyperparams
    max_iter_var: int = 500,
    learning_rate_var: float = 0.05,
    max_depth_var: int | None = 6,
):
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a pandas DataFrame.")
    X = X_df.reset_index(drop=True).copy()

    A = np.asarray(A_df).reshape(-1)
    A = np.rint(A).astype(int)
    Y = np.asarray(Y_df, dtype=float).reshape(-1)

    if len(X) != len(A) or len(X) != len(Y):
        raise ValueError("Length mismatch among X_df, A_df, Y_df.")
    if not set(np.unique(A)).issubset({0, 1}):
        raise ValueError("A must be binary in {0,1}.")

    X_all = X.copy()
    X_all["A"] = A

    mean_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(learning_rate_mean),
        max_iter=int(max_iter_mean),
        max_depth=max_depth_mean,
        random_state=int(seed),
    )

    # --- OOF mean preds for residuals ---
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    mu_oof = np.empty_like(Y, dtype=float)

    for tr, te in kf.split(X_all):
        m = clone(mean_model)
        m.fit(X_all.iloc[tr], Y[tr])
        mu_oof[te] = m.predict(X_all.iloc[te])

    resid_oof = Y - mu_oof

    # stabilize log(resid^2)
    eps = 1e-6 * np.var(resid_oof) + 1e-12
    log_r2 = np.log(resid_oof**2 + eps)

    var_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(learning_rate_var),
        max_iter=int(max_iter_var),
        max_depth=max_depth_var,
        random_state=int(seed) + 1,
    )
    var_model.fit(X_all, log_r2)

    # --- variance scaling calibration ---
    log_r2_hat = var_model.predict(X_all)
    sigma_hat = np.sqrt(np.exp(log_r2_hat))
    c = np.sqrt(np.mean(resid_oof**2) / (np.mean(sigma_hat**2) + 1e-12))
    var_model.sigma_scale_ = float(c)

    # --- refit mean on full data ---
    mean_model.fit(X_all, Y)
    return mean_model, var_model


def fit_nn_mean_and_variance_cv(
    X_df: pd.DataFrame,
    A_df,
    Y_df,
    *,
    seed: int = 42,
    n_splits: int = 5,
    hidden_dims=(64, 32),
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 400,
    patience: int = 25,
    min_delta: float = 1e-3,
    validation_fraction: float = 0.1,
):
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a pandas DataFrame.")
    X = X_df.reset_index(drop=True).copy()

    A = np.asarray(A_df).reshape(-1)
    A = np.rint(A).astype(int)
    Y = np.asarray(Y_df, dtype=float).reshape(-1)

    if len(X) != len(A) or len(X) != len(Y):
        raise ValueError("Length mismatch among X_df, A_df, Y_df.")
    if not set(np.unique(A)).issubset({0, 1}):
        raise ValueError("A must be binary in {0,1}.")

    X_all = X.copy()
    X_all["A"] = A

    def _make_mlp(local_seed: int):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(int(v) for v in hidden_dims),
                        activation="relu",
                        solver="adam",
                        alpha=float(weight_decay),
                        batch_size=int(max(16, batch_size)),
                        learning_rate_init=float(lr),
                        max_iter=int(max_epochs),
                        shuffle=True,
                        random_state=int(local_seed),
                        early_stopping=True,
                        validation_fraction=float(validation_fraction),
                        n_iter_no_change=int(patience),
                        tol=float(min_delta),
                    ),
                ),
            ]
        )

    mean_model = _make_mlp(int(seed))

    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    mu_oof = np.empty_like(Y, dtype=float)

    for tr, te in kf.split(X_all):
        m = clone(mean_model)
        m.fit(X_all.iloc[tr], Y[tr])
        mu_oof[te] = m.predict(X_all.iloc[te])

    resid_oof = Y - mu_oof
    eps = 1e-6 * np.var(resid_oof) + 1e-12
    log_r2 = np.log(resid_oof**2 + eps)

    var_model = _make_mlp(int(seed) + 1)
    var_model.fit(X_all, log_r2)

    log_r2_hat = var_model.predict(X_all)
    sigma_hat = np.sqrt(np.exp(log_r2_hat))
    c = np.sqrt(np.mean(resid_oof**2) / (np.mean(sigma_hat**2) + 1e-12))
    setattr(var_model, "sigma_scale_", float(c))

    mean_model.fit(X_all, Y)
    return mean_model, var_model


def fit_ols_mean_and_variance_poly(
    X_df: pd.DataFrame,
    A_df,
    Y_df,
    *,
    n_splits: int = 5,
    degree: int = 2,
    include_bias: bool = False,
):
    """
    Mean: OLS on (X, A) with all degree-2 terms (squares + pairwise interactions).
    Variance: OOF residuals -> log(resid^2) -> OLS on the same expanded design.
    """
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df must be a pandas DataFrame.")

    X = X_df.reset_index(drop=True).copy()
    A = np.asarray(A_df).reshape(-1)
    A = np.rint(A).astype(int)
    Y = np.asarray(Y_df, dtype=float).reshape(-1)

    if len(X) != len(A) or len(X) != len(Y):
        raise ValueError("Length mismatch among X_df, A_df, Y_df.")
    if not set(np.unique(A)).issubset({0, 1}):
        raise ValueError("A must be binary in {0,1}.")

    X_all = X.copy()
    X_all["A"] = A

    poly = PolynomialFeatures(degree=int(degree), include_bias=bool(include_bias))
    Z_all = poly.fit_transform(X_all)

    mean_model = LinearRegression()
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=42)
    mu_oof = np.empty_like(Y, dtype=float)

    for tr, te in kf.split(Z_all):
        m = clone(mean_model)
        m.fit(Z_all[tr], Y[tr])
        mu_oof[te] = m.predict(Z_all[te])

    resid_oof = Y - mu_oof
    eps = 1e-6 * np.var(resid_oof) + 1e-12
    log_r2 = np.log(resid_oof**2 + eps)

    var_model = LinearRegression()
    var_model.fit(Z_all, log_r2)

    log_r2_hat = var_model.predict(Z_all)
    sigma_hat = np.sqrt(np.exp(log_r2_hat))
    c = np.sqrt(np.mean(resid_oof**2) / (np.mean(sigma_hat**2) + 1e-12))
    var_model.sigma_scale_ = float(c)

    mean_model.fit(Z_all, Y)

    return {
        "poly": poly,
        "mean_model": mean_model,
        "var_model": var_model,
        "feature_names_in_": list(X_all.columns),
        "degree": int(degree),
        "include_bias": bool(include_bias),
    }


def cache_ols_mu_sigma(ols_bundle, X_df: pd.DataFrame):
    X_base = X_df.reset_index(drop=True).copy()
    X0 = X_base.copy(); X0["A"] = 0
    X1 = X_base.copy(); X1["A"] = 1

    poly = ols_bundle["poly"]
    mean_model = ols_bundle["mean_model"]
    var_model = ols_bundle["var_model"]

    Z0 = poly.transform(X0)
    Z1 = poly.transform(X1)

    mu0 = np.asarray(mean_model.predict(Z0), dtype=float).reshape(-1)
    mu1 = np.asarray(mean_model.predict(Z1), dtype=float).reshape(-1)

    logr2_0 = np.asarray(var_model.predict(Z0), dtype=float).reshape(-1)
    logr2_1 = np.asarray(var_model.predict(Z1), dtype=float).reshape(-1)

    scale = getattr(var_model, "sigma_scale_", 1.0)
    sigma0 = np.sqrt(np.exp(logr2_0)) * float(scale)
    sigma1 = np.sqrt(np.exp(logr2_1)) * float(scale)

    n = len(X_df)
    if mu0.shape != (n,) or mu1.shape != (n,):
        raise ValueError(f"Bad mu shapes: mu0 {mu0.shape}, mu1 {mu1.shape}, expected ({n},)")
    if sigma0.shape != (n,) or sigma1.shape != (n,):
        raise ValueError(f"Bad sigma shapes: s0 {sigma0.shape}, s1 {sigma1.shape}, expected ({n},)")

    return mu0, mu1, sigma0, sigma1


def cache_gb_mu_sigma(mean_model, var_model, X_df: pd.DataFrame):
    X_base = X_df.reset_index(drop=True).copy()
    X0 = X_base.copy(); X0["A"] = 0
    X1 = X_base.copy(); X1["A"] = 1

    mu0 = np.asarray(mean_model.predict(X0), dtype=float).reshape(-1)
    mu1 = np.asarray(mean_model.predict(X1), dtype=float).reshape(-1)

    logr2_0 = np.asarray(var_model.predict(X0), dtype=float).reshape(-1)
    logr2_1 = np.asarray(var_model.predict(X1), dtype=float).reshape(-1)

    scale = getattr(var_model, "sigma_scale_", 1.0)
    sigma0 = np.sqrt(np.exp(logr2_0)) * float(scale)
    sigma1 = np.sqrt(np.exp(logr2_1)) * float(scale)

    n = len(X_df)
    if mu0.shape != (n,) or mu1.shape != (n,):
        raise ValueError(f"Bad mu shapes: mu0 {mu0.shape}, mu1 {mu1.shape}, expected ({n},)")
    if sigma0.shape != (n,) or sigma1.shape != (n,):
        raise ValueError(f"Bad sigma shapes: s0 {sigma0.shape}, s1 {sigma1.shape}, expected ({n},)")

    return mu0, mu1, sigma0, sigma1


def fit_rf_mean_and_variance_oob(
    X_df: pd.DataFrame,
    A_df,
    Y_df,
    *,
    seed: int = 42,
    # mean model hyperparams
    rf_ntrees: int = 800,
    rf_minleaf: int = 1,
    rf_maxfeat=1.0,
    rf_bootstrap: bool = True,
    rf_max_depth=None,
    rf_n_jobs: int = 1,
):
    """
    Mean: single RF on (X,A)
    Variance: OOB residuals -> log(resid^2) -> second RF on (X,A)
    (This mirrors the "mean + log-variance" idea used for GB, but with RF.)
    """
    X = X_df.reset_index(drop=True).copy()
    A = np.asarray(A_df).reshape(-1)
    A = np.rint(A).astype(int)
    Y = np.asarray(Y_df, dtype=float).reshape(-1)

    X_all = X.copy()
    X_all["A"] = A

    mean_model = RandomForestRegressor(
        n_estimators=int(rf_ntrees),
        min_samples_leaf=int(rf_minleaf),
        max_features=rf_maxfeat,
        bootstrap=bool(rf_bootstrap),
        oob_score=bool(rf_bootstrap),
        max_depth=rf_max_depth,
        n_jobs=int(rf_n_jobs),
        random_state=int(seed),
    )
    mean_model.fit(X_all, Y)

    # OOB predictions (only available if bootstrap=True and oob_score=True)
    if not hasattr(mean_model, "oob_prediction_") or mean_model.oob_prediction_ is None:
        raise ValueError("RF OOB predictions unavailable. Ensure rf_bootstrap=True.")

    mu_oob = np.asarray(mean_model.oob_prediction_, dtype=float).reshape(-1)
    resid_oob = Y - mu_oob

    eps = 1e-6 * np.var(resid_oob) + 1e-12
    log_r2 = np.log(resid_oob**2 + eps)

    var_model = RandomForestRegressor(
        n_estimators=int(rf_ntrees),
        min_samples_leaf=int(rf_minleaf),
        max_features=rf_maxfeat,
        bootstrap=bool(rf_bootstrap),
        max_depth=rf_max_depth,
        n_jobs=int(rf_n_jobs),
        random_state=int(seed) + 1,
    )
    var_model.fit(X_all, log_r2)

    # calibration (same spirit as GB block)
    log_r2_hat = var_model.predict(X_all)
    sigma_hat = np.sqrt(np.exp(log_r2_hat))
    c = np.sqrt(np.mean(resid_oob**2) / (np.mean(sigma_hat**2) + 1e-12))
    var_model.sigma_scale_ = float(c)

    return mean_model, var_model


def cache_rf_mu_sigma(mean_model, var_model, X_df: pd.DataFrame):
    # same caching layout as GB (so the generator interface stays identical)
    X_base = X_df.reset_index(drop=True).copy()
    X0 = X_base.copy(); X0["A"] = 0
    X1 = X_base.copy(); X1["A"] = 1

    mu0 = np.asarray(mean_model.predict(X0), dtype=float).reshape(-1)
    mu1 = np.asarray(mean_model.predict(X1), dtype=float).reshape(-1)

    logr2_0 = np.asarray(var_model.predict(X0), dtype=float).reshape(-1)
    logr2_1 = np.asarray(var_model.predict(X1), dtype=float).reshape(-1)

    scale = getattr(var_model, "sigma_scale_", 1.0)
    sigma0 = np.sqrt(np.exp(logr2_0)) * float(scale)
    sigma1 = np.sqrt(np.exp(logr2_1)) * float(scale)

    n = len(X_df)
    if mu0.shape != (n,) or mu1.shape != (n,):
        raise ValueError(f"Bad mu shapes: mu0 {mu0.shape}, mu1 {mu1.shape}, expected ({n},)")
    if sigma0.shape != (n,) or sigma1.shape != (n,):
        raise ValueError(f"Bad sigma shapes: s0 {sigma0.shape}, s1 {sigma1.shape}, expected ({n},)")

    return mu0, mu1, sigma0, sigma1


def cache_nn_mu_sigma(mean_model, var_model, X_df: pd.DataFrame):
    X_base = X_df.reset_index(drop=True).copy()
    X0 = X_base.copy(); X0["A"] = 0
    X1 = X_base.copy(); X1["A"] = 1

    mu0 = np.asarray(mean_model.predict(X0), dtype=float).reshape(-1)
    mu1 = np.asarray(mean_model.predict(X1), dtype=float).reshape(-1)

    logr2_0 = np.asarray(var_model.predict(X0), dtype=float).reshape(-1)
    logr2_1 = np.asarray(var_model.predict(X1), dtype=float).reshape(-1)

    scale = getattr(var_model, "sigma_scale_", 1.0)
    sigma0 = np.sqrt(np.exp(logr2_0)) * float(scale)
    sigma1 = np.sqrt(np.exp(logr2_1)) * float(scale)

    n = len(X_df)
    if mu0.shape != (n,) or mu1.shape != (n,):
        raise ValueError(f"Bad mu shapes: mu0 {mu0.shape}, mu1 {mu1.shape}, expected ({n},)")
    if sigma0.shape != (n,) or sigma1.shape != (n,):
        raise ValueError(f"Bad sigma shapes: s0 {sigma0.shape}, s1 {sigma1.shape}, expected ({n},)")

    return mu0, mu1, sigma0, sigma1
