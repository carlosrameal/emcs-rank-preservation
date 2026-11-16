# mc_workers.py
from __future__ import annotations
import os
import inspect
import traceback
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance
import estimators as estimator_lib

_G: dict = {}
_TP_LIMITS = None


def _set_thread_limits_worker(t: int):
    global _TP_LIMITS
    t = int(max(1, t))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["MKL_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = str(t)
    os.environ["NUMEXPR_NUM_THREADS"] = str(t)
    try:
        from threadpoolctl import threadpool_limits
        _TP_LIMITS = threadpool_limits(limits=t)
    except Exception:
        _TP_LIMITS = None


def _filter_kwargs_for_fn_local(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        accepted = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        return {}


def _js_divergence_hist(x, y, bins="fd"):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    hist_x, bin_edges = np.histogram(x, bins=bins, density=True)
    hist_y, _ = np.histogram(y, bins=bin_edges, density=True)

    px = hist_x / (hist_x.sum() + 1e-12)
    py = hist_y / (hist_y.sum() + 1e-12)
    return float(jensenshannon(px, py, base=2.0) ** 2)


def _distribution_distances_local(y_gen, y_ref, A_np, label):
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
            "JS_hist": float(_js_divergence_hist(yr, yg, bins="fd")),
            "MeanDiff": float(np.mean(yg) - np.mean(yr)),
            "StdRatio": float(np.std(yg, ddof=1) / (np.std(yr, ddof=1) + 1e-12)),
        })
    return rows


def _estimator_runtime_kwargs(est_label: str) -> dict:
    kwargs = dict(_G["estimator_params"].get(est_label, {}))
    for bad in ("seed", "random_state", "alpha"):
        kwargs.pop(bad, None)

    if est_label in ("RF_GCOMP", "GB_RF_AIPW", "GB_RF_TMLE"):
        kwargs.setdefault("rf_n_jobs", int(_G["threads_per_proc"]))
        kwargs.setdefault("n_jobs", int(_G["threads_per_proc"]))
    if est_label == "BCF_GCOMP":
        kwargs.setdefault("num_threads", int(_G["threads_per_proc"]))
    return kwargs


def _prepare_shared_outcome_fits(seed: int, y: np.ndarray, a_rep: np.ndarray) -> dict:
    shared_fits = {}
    for est_label in _G["estimators"].keys():
        runtime_kwargs = _estimator_runtime_kwargs(est_label)
        fit_key = estimator_lib.shared_outcome_fit_key(est_label, runtime_kwargs)
        if fit_key is None or fit_key in shared_fits:
            continue
        shared_fits[fit_key] = estimator_lib.fit_shared_outcome_for_estimator(
            est_label,
            _G["X_df"],
            a_rep,
            y,
            seed=seed,
            **runtime_kwargs,
        )
    return shared_fits


def _prepare_shared_propensity_fits(seed: int, a_rep: np.ndarray) -> dict:
    shared_fits = {}
    for est_label in _G["estimators"].keys():
        runtime_kwargs = _estimator_runtime_kwargs(est_label)
        fit_key = estimator_lib.shared_propensity_fit_key(est_label, runtime_kwargs)
        if fit_key is None or fit_key in shared_fits:
            continue
        shared_fits[fit_key] = estimator_lib.fit_shared_propensity_for_estimator(
            est_label,
            _G["X_df"],
            a_rep,
            seed=seed,
            **runtime_kwargs,
        )
    return shared_fits


def _run_estimators_for_generated_y(r: int, seed: int, y: np.ndarray, a_rep: np.ndarray):
    dist_rows = _distribution_distances_local(
        y_gen=y,
        y_ref=_G["y_ref"],
        A_np=a_rep,
        label=_G["gen_label"],
    )

    shared_outcome_fits = _prepare_shared_outcome_fits(seed=seed, y=y, a_rep=a_rep)
    shared_propensity_fits = _prepare_shared_propensity_fits(seed=seed, a_rep=a_rep)
    est_rows = []
    for est_label, est_fn in _G["estimators"].items():
        kwargs = _estimator_runtime_kwargs(est_label)
        fit_key = estimator_lib.shared_outcome_fit_key(est_label, kwargs)
        if fit_key is not None and fit_key in shared_outcome_fits:
            kwargs["shared_outcome_fit"] = shared_outcome_fits[fit_key]
        ps_fit_key = estimator_lib.shared_propensity_fit_key(est_label, kwargs)
        if ps_fit_key is not None and ps_fit_key in shared_propensity_fits:
            kwargs["shared_propensity_fit"] = shared_propensity_fits[ps_fit_key]

        kwargs = _filter_kwargs_for_fn_local(est_fn, kwargs)
        try:
            ate_hat, se_hat, ci_lo, ci_hi, cover = est_fn(
                _G["X_df"], a_rep, y,
                true_ate=_G["true_ate"],
                seed=seed,
                alpha=_G["alpha"],
                **kwargs
            )
        except Exception:
            print(
                f"[mc worker pid={os.getpid()}] estimator failure | "
                f"gen={_G['gen_label']} | est={est_label} | rep={r} | seed={seed}",
                flush=True,
            )
            traceback.print_exc()
            raise
        est_rows.append((r, seed, est_label, ate_hat, se_hat, ci_lo, ci_hi, cover))

    return est_rows, dist_rows


def init_chunk_worker(
    gen_label: str,
    y_ref: np.ndarray,
    true_ate: float,
    estimators: dict,
    estimator_params: dict,
    seed_base: int,
    alpha: float,
    threads_per_proc: int,
    X_df: pd.DataFrame,
    A_np: np.ndarray,
    A_draws: np.ndarray | None,
    y_draws: np.ndarray,
    progress_queue,
    progress_every: int,
):
    _G["gen_label"] = str(gen_label)
    _G["y_ref"] = np.asarray(y_ref, dtype=float).reshape(-1)
    _G["true_ate"] = float(true_ate)
    _G["estimators"] = estimators
    _G["estimator_params"] = estimator_params or {}
    _G["seed_base"] = int(seed_base)
    _G["alpha"] = float(alpha)
    _G["threads_per_proc"] = int(threads_per_proc)
    _G["X_df"] = X_df
    _G["A_np"] = np.asarray(A_np).reshape(-1).astype(int)
    _G["A_draws"] = None if A_draws is None else np.asarray(A_draws).astype(int)
    _G["y_draws"] = np.asarray(y_draws, dtype=np.float32)
    _G["progress_queue"] = progress_queue
    _G["progress_every"] = int(max(1, progress_every))

    _set_thread_limits_worker(int(_G["threads_per_proc"]))


def do_rep_chunk(rep_indices):
    _set_thread_limits_worker(int(_G["threads_per_proc"]))

    rep_indices = [int(r) for r in rep_indices]
    est_rows_all = []
    dist_rows_all = []
    processed_since_report = 0

    for r in rep_indices:
        seed = int(_G["seed_base"] + r)
        y = np.asarray(_G["y_draws"][:, r], dtype=float).reshape(-1)
        if _G.get("A_draws") is None:
            a_rep = np.asarray(_G["A_np"], dtype=int).reshape(-1)
        else:
            a_rep = np.asarray(_G["A_draws"][:, r], dtype=int).reshape(-1)
        est_rows, dist_rows = _run_estimators_for_generated_y(r=r, seed=seed, y=y, a_rep=a_rep)
        est_rows_all.extend(est_rows)
        dist_rows_all.extend(dist_rows)
        processed_since_report += 1

        progress_queue = _G.get("progress_queue")
        progress_every = int(_G.get("progress_every", 1))
        if progress_queue is not None and processed_since_report >= progress_every:
            progress_queue.put(processed_since_report)
            processed_since_report = 0

    progress_queue = _G.get("progress_queue")
    if progress_queue is not None and processed_since_report > 0:
        progress_queue.put(processed_since_report)

    return {
        "rep_indices": rep_indices,
        "n_reps": len(rep_indices),
        "est_rows": est_rows_all,
        "dist_rows": dist_rows_all,
    }
