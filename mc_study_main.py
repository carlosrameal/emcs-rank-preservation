from __future__ import annotations

# mc_study_main.py
# ==============================================================================
# Driver script for the Monte Carlo study on a generic observed dataset with
# columns arranged as (X, A, Y): covariates, binary treatment, and outcome.
# ==============================================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
from datetime import datetime
from pathlib import Path
import time

import pandas as pd
from structured_dgps import (
    build_structured_dgp_label,
    generate_structured_observed_data,
    is_nie_dgp,
    resolve_default_data_path,
    selected_dgp_code,
    structured_mode_uses_shared_cache,
)

print(
    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
    "mc_study_main.py bootstrap started",
    flush=True,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--outcome-col", type=str, default=None)
    parser.add_argument("--treatment-col", type=str, default="A")
    parser.add_argument("--job-seed", type=int, default=42)
    parser.add_argument(
        "--dgp",
        type=str,
        default=None,
        choices=["NieA", "NieB", "NieC", "NieD"],
        help="Purely synthetic structured DGP. Omit to retain the legacy ACIC selector.",
    )
    parser.add_argument(
        "--acic-error",
        type=str,
        default="iid",
        choices=["iid", "group_corr", "heteroskedastic", "nonadditive"],
    )
    parser.add_argument("--acic-magnitude", type=int, default=1)
    parser.add_argument("--acic-noise", type=int, default=1)
    parser.add_argument("--acic-confounding", type=int, default=1)
    parser.add_argument("--acic-seed", type=int, default=None)
    parser.add_argument(
        "--acic-treatment-mode",
        type=str,
        default="fixed",
        choices=["fixed", "redraw"],
    )
    parser.add_argument("--synthetic-n", type=int, default=1000)
    parser.add_argument("--synthetic-d", type=int, default=6)
    parser.add_argument("--synthetic-sigma", type=float, default=1.0)
    parser.add_argument("--drop-cols", nargs="*", default=[])
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--shared-output-root", type=str, default=None)
    parser.add_argument("--full-output", type=_as_bool, default=True)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--job-submit-epoch", type=float, default=None)
    parser.add_argument("--requested-cpus", type=int, default=None)
    parser.add_argument("--dgp-label", type=str, default=None)
    parser.add_argument("--dgp-truth-path", type=str, default=None)
    parser.add_argument("--bivar-cols", nargs="*", default=None)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["OLS", "GB", "RF", "NN", "WGAN"],
        choices=["OLS", "GB", "RF", "NN", "WGAN"],
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["OLS1", "OLS2_GCOMP", "GB_IPW", "GB_GCOMP", "RF_GCOMP", "GB_GB_AIPW", "GB_RF_AIPW", "GB_OLS2_AIPW", "GB_GB_TMLE", "GB_RF_TMLE", "GB_OLS2_TMLE", "BCF_GCOMP", "NN_GCOMP", "GB_NN_AIPW", "GB_NN_TMLE"],
        choices=["OLS1", "OLS2", "OLS2_GCOMP", "GB_IPW", "GB_GCOMP", "RF_GCOMP", "GB_GB_AIPW", "GB_RF_AIPW", "GB_OLS2_AIPW", "GB_GB_TMLE", "GB_RF_TMLE", "GB_OLS2_TMLE", "GB_NN_TMLE", "BCF_GCOMP", "NN_GCOMP", "GB_NN_AIPW"],
    )
    parser.add_argument("--n-mc", type=int, default=1000)
    parser.add_argument("--dgp-benchmark-n-mc", type=int, default=None)
    parser.add_argument(
        "--skip-real-dgp-benchmark",
        action="store_true",
        help=(
            "Do not run or load the REAL_DGP estimator benchmark. Intended for "
            "a simulator-only rerun whose output will be merged with an existing "
            "per-seed REAL_DGP benchmark."
        ),
    )
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--wgan-epochs", type=int, default=5000)
    parser.add_argument("--wgan-batch-size", type=int, default=512)
    parser.add_argument("--wgan-print-every", type=int, default=200)
    return parser.parse_args()


def read_table(data_path: str | Path) -> pd.DataFrame:
    import pandas as pd

    path = Path(data_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists():
        suffix = path.suffix.lower()
        if suffix in {"", ".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported data format: {suffix}")

    parquet_fallback = Path(str(path) + ".parquet")
    if parquet_fallback.exists():
        return pd.read_parquet(parquet_fallback)

    csv_fallback = Path(str(path) + ".csv")
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback)

    raise FileNotFoundError(f"Could not find data file '{data_path}'")


def choose_bivariates(X_df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        return [c for c in requested if c in X_df.columns][:3]
    return X_df.columns.tolist()[:3]




def build_seed_plan(*, job_seed: int, acic_seed_override: int | None) -> dict[str, int]:
    base = int(job_seed)
    acic_base = int(acic_seed_override) if acic_seed_override is not None else base
    return {
        "job_seed": base,
        "acic_treatment": acic_base,
        "acic_outcome": acic_base + 1,
        "propensity_model": 54_321,
        "ols_diag_fit": base + 201,
        "ols_diag_sample": base + 202,
        "gb_fit": base + 301,
        "gb_diag_fit": base + 302,
        "gb_diag_sample": base + 303,
        "rf_fit": base + 401,
        "rf_diag_fit": base + 402,
        "rf_diag_sample": base + 403,
        "nn_fit": base + 431,
        "nn_diag_fit": base + 432,
        "nn_diag_sample": base + 433,
        "wgan_init": base + 500,
        "wgan_fit": base + 501,
        "wgan_diag_fit": base + 502,
        "wgan_diag_sample": base + 503,
        "mc_base": base + 100_000,
    }


def canonicalize_estimator_names(estimator_names: list[str]) -> list[str]:
    canonical = [str(name) for name in estimator_names]
    return list(dict.fromkeys(canonical))


def build_dgp_label(args, *, data_path: str | Path) -> str:
    if args.dgp_label:
        return str(args.dgp_label)

    stem = Path(data_path).stem or "dataset"
    if args.outcome_col is None:
        label = (
            f"{stem}__{args.acic_error}"
            f"__mag{int(args.acic_magnitude)}"
            f"__noise{int(args.acic_noise)}"
            f"__conf{int(args.acic_confounding)}"
        )
        if str(args.acic_treatment_mode) == "redraw":
            label += "__tredraw"
        return label
    return stem


def resolve_dgp_truth_path(
    *,
    out_root: Path,
    dgp_label: str,
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p
    return out_root / "tracked" / "_dgp_truth" / f"{dgp_label}__truth_metrics.npz"


def resolve_dgp_benchmark_dirs(
    *,
    out_root: Path,
    dgp_label: str,
    full_output: bool,
) -> tuple[Path | None, Path]:
    if full_output:
        tracked_dir = out_root / "tracked" / "_dgp_benchmark" / dgp_label
        untracked_dir = out_root / "untracked" / "_dgp_benchmark" / dgp_label
        return tracked_dir, untracked_dir
    tracked_dir = out_root / "csv_only" / "_dgp_benchmark" / dgp_label
    untracked_dir = tracked_dir
    return tracked_dir, untracked_dir


def resolve_job_local_dgp_truth_path(
    *,
    out_root: Path,
    dgp_label: str,
) -> Path:
    return out_root / "_dgp_truth" / f"{dgp_label}__truth_metrics.npz"


def resolve_job_local_dgp_benchmark_dirs(
    *,
    out_root: Path,
    dgp_label: str,
) -> tuple[Path, Path]:
    local_dir = out_root / "_dgp_benchmark" / dgp_label
    return local_dir, local_dir


def resolve_run_output_dirs(
    *,
    out_root: Path,
    dgp_label: str,
    run_label: str | None,
    run_id: str | None,
    full_output: bool,
) -> tuple[Path | None, Path]:
    if full_output:
        if run_label:
            tracked_dir = out_root / "tracked" / run_label / "custom_selection"
            untracked_dir = out_root / "untracked" / run_label / "custom_selection"
        else:
            tracked_dir = out_root / "tracked" / "custom_selection"
            untracked_dir = out_root / "untracked" / "custom_selection"
        return tracked_dir, untracked_dir

    return out_root, out_root


def save_dgp_truth_metrics(
    path: Path,
    *,
    dgp_label: str,
    cate_true,
    ate_true: float,
    metadata: dict,
):
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cate_true=np.asarray(cate_true, dtype=float).reshape(-1),
        ate_true=np.asarray([float(ate_true)], dtype=float),
        dgp_label=np.asarray([str(dgp_label)]),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True)]),
    )


def load_dgp_truth_metrics(path: Path) -> dict:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        cate_true = np.asarray(data["cate_true"], dtype=float).reshape(-1)
        ate_true = float(np.asarray(data["ate_true"], dtype=float).reshape(-1)[0])
        dgp_label = str(np.asarray(data["dgp_label"]).reshape(-1)[0])
        metadata_raw = str(np.asarray(data["metadata_json"]).reshape(-1)[0])
    return {
        "cate_true": cate_true,
        "ate_true": ate_true,
        "dgp_label": dgp_label,
        "metadata": json.loads(metadata_raw),
    }


def stable_seed_from_text(text: str) -> int:
    import hashlib

    raw = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % 2_000_000_000


def parse_run_id_submit_epoch(run_id: str | None) -> float | None:
    if not run_id:
        return None
    run_id = str(run_id).strip()
    if len(run_id) < 15:
        return None
    try:
        dt = datetime.strptime(run_id[:15], "%Y%m%d_%H%M%S")
        return float(dt.timestamp())
    except Exception:
        return None


def load_perf_summary(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path)


def dgp_benchmark_is_usable(
    perf_df: pd.DataFrame,
    *,
    expected_estimators: list[str],
    expected_n_mc: int,
) -> tuple[bool, str]:
    import pandas as pd

    if perf_df is None or perf_df.empty:
        return False, "summary is empty"
    required_cols = {"generator", "estimator"}
    if not required_cols.issubset(perf_df.columns):
        return False, f"missing required columns: {sorted(required_cols - set(perf_df.columns))}"

    real_df = perf_df.loc[perf_df["generator"].astype(str) == "REAL_DGP"].copy()
    if real_df.empty:
        return False, "REAL_DGP rows are missing"

    present_estimators = set(real_df["estimator"].astype(str).tolist())
    missing_estimators = [e for e in expected_estimators if e not in present_estimators]
    if missing_estimators:
        return False, f"missing estimator rows: {missing_estimators}"

    if "N_MC" not in real_df.columns:
        return False, "N_MC column is missing"

    n_vals = pd.to_numeric(real_df["N_MC"], errors="coerce").dropna().astype(int).unique().tolist()
    if n_vals != [int(expected_n_mc)]:
        return False, f"N_MC mismatch: found {n_vals}, expected {[int(expected_n_mc)]}"

    return True, "ok"


def rank_generator_table(perf_df: pd.DataFrame, *, generator: str) -> pd.DataFrame:
    import numpy as np

    g = perf_df.loc[perf_df["generator"].astype(str) == str(generator)].copy()
    if g.empty:
        return g

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
    return g[cols]


def print_ranked_generator_table(
    perf_df: pd.DataFrame,
    *,
    generator: str,
    heading: str,
):
    ranked = rank_generator_table(perf_df, generator=generator)
    print(heading)
    if ranked.empty:
        print("  (no rows)")
        return
    print(ranked.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main():
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        "mc_study_main.py started - loading base libraries...",
        flush=True,
    )

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier

    args = parse_args()
    if not args.data_path and not is_nie_dgp(args):
        args.data_path = str(
            resolve_default_data_path(
                project_root=PROJECT_ROOT,
            )
        )
    overall_start = time.time()
    requested_generators = list(dict.fromkeys(args.generators))
    requested_estimators = canonicalize_estimator_names(args.estimators)
    seed_plan = build_seed_plan(
        job_seed=int(args.job_seed),
        acic_seed_override=args.acic_seed,
    )

    df = read_table(args.data_path) if args.data_path else None
    run_label = str(args.run_label).strip() if args.run_label else None
    structured = None

    if args.outcome_col is None:
        structured = generate_structured_observed_data(
            df,
            args,
            seed_plan=seed_plan,
        )
        X_df = structured["X_df"]
        A_binary = np.asarray(structured["A_binary"], dtype=int).reshape(-1)
        Y_parts = structured["Y_parts"]
        Y_np = np.asarray(Y_parts["y"], dtype=float).reshape(-1)
        outcome_label = str(structured["outcome_label"])
    else:
        required_cols = {args.outcome_col, args.treatment_col}
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        drop_cols = [c for c in args.drop_cols if c in df.columns and c not in required_cols]
        X_df = df.drop(columns=[args.outcome_col, args.treatment_col, *drop_cols]).copy()
        A_df = df[args.treatment_col].copy()
        Y_df = df[args.outcome_col].copy()

        A_binary = np.rint(A_df.to_numpy(dtype=float).reshape(-1)).astype(int)
        if not set(np.unique(A_binary)).issubset({0, 1}):
            raise ValueError("Treatment column must be binary and coded as {0,1}.")

        Y_np = Y_df.to_numpy(dtype=float).reshape(-1)
        outcome_label = args.outcome_col

    A_np = A_binary.reshape(-1, 1).astype(float)
    bivar_cols = choose_bivariates(X_df, args.bivar_cols)

    out_root = Path(args.output_root)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    shared_out_root = Path(args.shared_output_root) if args.shared_output_root else out_root
    if not shared_out_root.is_absolute():
        shared_out_root = PROJECT_ROOT / shared_out_root
    if args.outcome_col is None:
        dgp_label = build_structured_dgp_label(args, data_path=args.data_path)
    else:
        dgp_label = build_dgp_label(args, data_path=args.data_path)
    full_output = bool(args.full_output)

    if full_output:
        if run_label:
            out_tracked = out_root / "tracked" / run_label
            out_untracked = out_root / "untracked" / run_label
        else:
            out_tracked = out_root / "tracked"
            out_untracked = out_root / "untracked"
        out_tracked.mkdir(parents=True, exist_ok=True)
        out_untracked.mkdir(parents=True, exist_ok=True)
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        out_untracked = out_root
        out_tracked = out_root

    use_shared_dgp_cache = (
        structured_mode_uses_shared_cache(args)
        if args.outcome_col is None else True
    )
    dgp_store_root = shared_out_root if use_shared_dgp_cache else out_root
    if use_shared_dgp_cache:
        dgp_truth_path = resolve_dgp_truth_path(
            out_root=dgp_store_root,
            dgp_label=dgp_label,
            explicit_path=args.dgp_truth_path,
        )
        dgp_benchmark_tracked_dir, dgp_benchmark_untracked_dir = resolve_dgp_benchmark_dirs(
            out_root=dgp_store_root,
            dgp_label=dgp_label,
            full_output=full_output,
        )
    else:
        dgp_truth_path = resolve_job_local_dgp_truth_path(
            out_root=dgp_store_root,
            dgp_label=dgp_label,
        )
        dgp_benchmark_tracked_dir, dgp_benchmark_untracked_dir = resolve_job_local_dgp_benchmark_dirs(
            out_root=dgp_store_root,
            dgp_label=dgp_label,
        )

    import_start = time.time()
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        "Loading study modules and dependencies...",
        flush=True,
    )
    import estimators as est
    from generators import GBGenerator, NNGenerator, OLSGenerator, RFGenerator, WGANGenerator
    from helpers import (
        cache_gb_mu_sigma,
        cache_nn_mu_sigma,
        cache_ols_mu_sigma,
        cache_rf_mu_sigma,
        fit_gb_mean_and_variance_cv,
        fit_nn_mean_and_variance_cv,
        fit_ols_mean_and_variance_poly,
        fit_rf_mean_and_variance_oob,
        run_cached_generator_diagnostics,
        run_wgan_generator_diagnostics,
    )
    from mc_run import run_mc_study, write_master_report
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Study modules loaded in {time.time() - import_start:.2f}s",
        flush=True,
    )
    preload_t0 = time.time()
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        "Preloading estimator dependencies in main process...",
        flush=True,
    )
    est.preload_estimator_dependencies(requested_estimators)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Estimator dependencies loaded in {time.time() - preload_t0:.2f}s",
        flush=True,
    )

    print("Data loaded successfully")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data path:   {Path(args.data_path).resolve() if args.data_path else '(generated in memory)'}")
    print(f"  Outcome:     {outcome_label}")
    print(f"  Treatment:   {args.treatment_col}")
    print(f"  X columns:   {X_df.shape[1]}")
    print(f"  N rows:      {len(X_df)}")
    print(f"  Run label:   {run_label if run_label else '(none)'}")
    print(f"  DGP label:   {dgp_label}")
    print(f"  Full output: {full_output}")
    print("Random seed plan:")
    print(f"  job_seed:            {seed_plan['job_seed']}")
    if args.outcome_col is None:
        print(f"  dgp:                 {selected_dgp_code(args)}")
        if not is_nie_dgp(args):
            print("  acic_version:        2017")
        if not is_nie_dgp(args) and str(args.acic_treatment_mode) == "redraw":
            print(f"  acic_treatment:      {seed_plan['acic_treatment']}")
        print(f"  observed_outcome:    {seed_plan['acic_outcome']}")
    print(f"  propensity_model:    {seed_plan['propensity_model']}")
    if "GB" in requested_generators:
        print(f"  gb_fit:              {seed_plan['gb_fit']}")
    if "RF" in requested_generators:
        print(f"  rf_fit:              {seed_plan['rf_fit']}")
    if "NN" in requested_generators:
        print(f"  nn_fit:              {seed_plan['nn_fit']}")
    if "WGAN" in requested_generators:
        print(f"  wgan_init:           {seed_plan['wgan_init']}")
        print(f"  wgan_fit:            {seed_plan['wgan_fit']}")
    print(f"  mc_base:             {seed_plan['mc_base']}")
    if args.outcome_col is None:
        print("  Outcome generation settings:")
        for line in structured["log_lines"]:
            print(line)
        if use_shared_dgp_cache:
            print("    benchmark_mc:     keep A fixed and redraw Y each replicate")
        else:
            print("    benchmark_mc:     job-local benchmark tied to the realized A for this job")
    print("Output directories:")
    print(f"  Job output root:  {out_root}")
    print(f"  Tracked:          {out_tracked if full_output else '(disabled)'}")
    print(f"  Untracked:        {out_untracked}")
    if use_shared_dgp_cache:
        print(f"  Shared DGP root:  {shared_out_root}")
    else:
        print(f"  Shared DGP root:  (disabled; job-local benchmark stored under {dgp_store_root})")
    submit_epoch = (
        float(args.job_submit_epoch)
        if args.job_submit_epoch is not None
        else parse_run_id_submit_epoch(args.run_id)
    )
    if args.run_id:
        print(f"  Run ID:           {args.run_id}")
    if args.requested_cpus is not None:
        print(f"  Requested CPUs:   {int(args.requested_cpus)}")
    if submit_epoch is not None:
        print(f"  Submit epoch:     {submit_epoch:.0f}")
    print("DGP truth artifact:")
    print(f"  Path:      {dgp_truth_path}")

    report_context = {
        "run_id": args.run_id,
        "requested_cpus": int(args.requested_cpus) if args.requested_cpus is not None else None,
        "job_submit_epoch": f"{submit_epoch:.0f}" if submit_epoch is not None else None,
        "job_seed": int(seed_plan["job_seed"]),
        "n_mc": int(args.n_mc),
        "full_output": bool(full_output),
        "dgp": selected_dgp_code(args) if args.outcome_col is None else None,
        "acic_version": "2017" if args.outcome_col is None and not is_nie_dgp(args) else None,
        "acic_treatment_mode": str(args.acic_treatment_mode) if args.outcome_col is None else None,
    }

    dgp_truth = None
    dgp_truth_source = None
    if dgp_truth_path.exists():
        dgp_truth = load_dgp_truth_metrics(dgp_truth_path)
        dgp_truth_source = "loaded"
    elif args.outcome_col is None:
        dgp_truth = {
            "cate_true": np.asarray(Y_parts["alpha"], dtype=float).reshape(-1),
            "ate_true": float(np.mean(np.asarray(Y_parts["alpha"], dtype=float).reshape(-1))),
            "dgp_label": dgp_label,
            "metadata": structured["metadata"],
        }
        save_dgp_truth_metrics(
            dgp_truth_path,
            dgp_label=dgp_truth["dgp_label"],
            cate_true=dgp_truth["cate_true"],
            ate_true=dgp_truth["ate_true"],
            metadata=dgp_truth["metadata"],
        )
        dgp_truth_source = "computed_and_saved"
    else:
        raise ValueError(
            "No DGP truth artifact found and no known generated-outcome DGP is available to compute one. "
            "Provide --dgp-truth-path or run in generated-outcome mode first."
        )

    if str(dgp_truth["dgp_label"]) != str(dgp_label):
        raise ValueError(
            f"DGP truth artifact label mismatch: expected '{dgp_label}' but found "
            f"'{dgp_truth['dgp_label']}' in {dgp_truth_path}"
        )
    if int(np.asarray(dgp_truth["cate_true"]).reshape(-1).shape[0]) != int(len(X_df)):
        raise ValueError(
            f"DGP truth artifact length mismatch: expected {len(X_df)} rows but found "
            f"{np.asarray(dgp_truth['cate_true']).reshape(-1).shape[0]} in {dgp_truth_path}"
        )

    print(f"  Source:     {dgp_truth_source}")
    print(f"  True ATE:   {float(dgp_truth['ate_true']):.6f}")
    if "WGAN" in requested_generators:
        from wgan_module import runtime_diagnostics

        rt = runtime_diagnostics()
        print("Runtime diagnostics:")
        print(f"  Available CPUs:       {rt['available_cpus']}")
        print(f"  CPU affinity:         {rt['cpu_affinity'] if rt['cpu_affinity'] is not None else 'unavailable'}")
        print(f"  CUDA available:       {rt['cuda_available']}")
        print(f"  CUDA device count:    {rt['cuda_device_count']}")
        print(f"  CUDA_VISIBLE_DEVICES: {rt['cuda_visible_devices']}")

    ps_seed = int(seed_plan["propensity_model"])
    trim = 0.01

    X_ps = X_df.to_numpy(dtype=float)
    ps_model = GradientBoostingClassifier(random_state=int(ps_seed))
    ps_model.fit(X_ps, A_binary)
    p1_hat = ps_model.predict_proba(X_ps)[:, 1].astype(float)
    p1_hat_trim = np.clip(p1_hat, trim, 1.0 - trim)
    print("Propensity score model fitted: GradientBoostingClassifier")
    print(f"  Trimming bounds: [{trim}, {1.0 - trim}]")

    noise = "normal"
    t_df = 7
    ols_shared_params = {
        "n_splits": 5,
        "degree": 2,
        "include_bias": False,
    }
    gb_shared_params = {
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
    }
    crossfit_shared_params = {
        "crossfit_n_splits": 5,
        "crossfit_stratified": True,
    }
    rf_shared_params = {
        "rf_ntrees": 800,
        "rf_minleaf": 1,
        "rf_maxfeat": 1.0,
        "rf_bootstrap": True,
        "rf_max_depth": None,
        "rf_n_jobs": 1,
    }
    nn_shared_params = {
        "hidden_dims": (64, 32),
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 400,
        "patience": 25,
        "min_delta": 1e-3,
        "validation_fraction": 0.1,
    }
    estimators = {
        "OLS1": est.ols,
        # Legacy in-sample quadratic OLS label, retained only to complete
        # existing archived benchmark runs.
        "OLS2": est.ols_poly_gcomp,
        "OLS2_GCOMP": est.ols_poly_gcomp,
        "GB_IPW": est.gb_ipw,
        "GB_GCOMP": est.gb_gcomp,
        "RF_GCOMP": est.rf_gcomp,
        "GB_GB_AIPW": est.gb_gb_aipw,
        "GB_RF_AIPW": est.gb_rf_aipw,
        "GB_OLS2_AIPW": est.gb_ols2_aipw,
        "GB_GB_TMLE": est.gb_gb_tmle,
        "GB_RF_TMLE": est.gb_rf_tmle,
        "GB_OLS2_TMLE": est.gb_ols2_tmle,
        "BCF_GCOMP": est.bcf_gcomp,
        "NN_GCOMP": est.nn_gcomp,
        "GB_NN_AIPW": est.gb_nn_aipw,
        "GB_NN_TMLE": est.gb_nn_tmle,
    }
    estimator_params = {
        "OLS1": dict(cov_type="HC3"),
        "OLS2": dict(se_method="none", **ols_shared_params),
        "OLS2_GCOMP": dict(se_method="none", **ols_shared_params, **crossfit_shared_params),
        "GB_IPW": dict(**crossfit_shared_params),
        "GB_GCOMP": dict(se_method="none", **gb_shared_params, **crossfit_shared_params),
        "RF_GCOMP": dict(se_method="none", K=10, **rf_shared_params, **crossfit_shared_params),
        "GB_GB_AIPW": dict(p1_fixed=p1_hat_trim, se_method="none", **gb_shared_params, **crossfit_shared_params),
        "GB_RF_AIPW": dict(p1_fixed=p1_hat_trim, se_method="none", **rf_shared_params, **crossfit_shared_params),
        "GB_OLS2_AIPW": dict(p1_fixed=p1_hat_trim, se_method="none", **ols_shared_params, **crossfit_shared_params),
        "GB_GB_TMLE": dict(p1_fixed=p1_hat_trim, se_method="none", **gb_shared_params, **crossfit_shared_params),
        "GB_RF_TMLE": dict(p1_fixed=p1_hat_trim, se_method="none", **rf_shared_params, **crossfit_shared_params),
        "GB_OLS2_TMLE": dict(p1_fixed=p1_hat_trim, se_method="none", **ols_shared_params, **crossfit_shared_params),
        "BCF_GCOMP": dict(p1_fixed=p1_hat_trim, se_method="none"),
        "NN_GCOMP": dict(se_method="none", **crossfit_shared_params),
        "GB_NN_AIPW": dict(p1_fixed=p1_hat_trim, se_method="none", **crossfit_shared_params),
        "GB_NN_TMLE": dict(p1_fixed=p1_hat_trim, se_method="none", **crossfit_shared_params),
    }
    estimators = {name: estimators[name] for name in requested_estimators if name in estimators}
    estimator_params = {name: estimator_params[name] for name in requested_estimators if name in estimator_params}

    dgp_benchmark_n_mc = int(args.dgp_benchmark_n_mc or args.n_mc)
    report_context["dgp_benchmark_n_mc"] = int(dgp_benchmark_n_mc)
    dgp_benchmark_report = (
        dgp_benchmark_tracked_dir / "MASTER_SUMMARY_REPORT.txt"
        if dgp_benchmark_tracked_dir is not None else None
    )
    dgp_benchmark_summary_csv = dgp_benchmark_untracked_dir / "MASTER_mc_summary.csv"
    dgp_benchmark_dist_csv = dgp_benchmark_untracked_dir / "MASTER_distribution_distances.csv"
    full_benchmark_untracked_dir = dgp_store_root / "untracked" / "_dgp_benchmark" / dgp_label
    fallback_benchmark_summary_csv = full_benchmark_untracked_dir / "MASTER_mc_summary.csv"
    fallback_benchmark_dist_csv = full_benchmark_untracked_dir / "MASTER_distribution_distances.csv"
    dgp_benchmark_seed_base = stable_seed_from_text(f"REAL_DGP::{dgp_label}")

    print("REAL_DGP benchmark artifact:")
    print(f"  Tracked dir:   {dgp_benchmark_tracked_dir}")
    print(f"  Untracked dir: {dgp_benchmark_untracked_dir}")
    print(f"  N_MC:          {dgp_benchmark_n_mc}")
    print(f"  Seed base:     {dgp_benchmark_seed_base}")

    dgp_benchmark_source = None
    dgp_perf_df = None
    dgp_dist_df = pd.DataFrame()
    dgp_needs_recompute_reason = None
    if args.skip_real_dgp_benchmark:
        dgp_benchmark_source = "skipped"
    elif use_shared_dgp_cache:
        existing_summary_csv = dgp_benchmark_summary_csv
        existing_dist_csv = dgp_benchmark_dist_csv
        if (not existing_summary_csv.exists()) and (not full_output) and fallback_benchmark_summary_csv.exists():
            existing_summary_csv = fallback_benchmark_summary_csv
            existing_dist_csv = fallback_benchmark_dist_csv

        if existing_summary_csv.exists():
            dgp_perf_df = load_perf_summary(existing_summary_csv)
            ok, reason = dgp_benchmark_is_usable(
                dgp_perf_df,
                expected_estimators=list(estimators.keys()),
                expected_n_mc=dgp_benchmark_n_mc,
            )
            if ok:
                dgp_benchmark_source = "loaded"
                if existing_dist_csv.exists():
                    dgp_dist_df = pd.read_csv(existing_dist_csv)
            else:
                dgp_perf_df = None
                dgp_needs_recompute_reason = reason

    if dgp_perf_df is None and not args.skip_real_dgp_benchmark:
        print("=" * 80)
        print("START OF THE REAL_DGP ESTIMATOR BENCHMARK")
        print("=" * 80)
        if not use_shared_dgp_cache:
            print("Job-local REAL_DGP benchmark required because the observed design is specific to this job.")
        elif dgp_needs_recompute_reason:
            print(f"Recomputing cached REAL_DGP benchmark because: {dgp_needs_recompute_reason}")
        else:
            print("No usable cached REAL_DGP benchmark found; computing it now.")

        if args.outcome_col is not None:
            raise ValueError(
                "REAL_DGP benchmark computation requires a known structured DGP. "
                "Run in generated-outcome mode or provide a precomputed benchmark artifact."
            )

        real_dgp_generator = structured["sampler"]

        dgp_perf_df, dgp_dist_df = run_mc_study(
            generators={"REAL_DGP": real_dgp_generator},
            y_ref=Y_np,
            true_ate_by_generator={"REAL_DGP": float(dgp_truth["ate_true"])},
            estimators=estimators,
            estimator_params=estimator_params,
            X_df=X_df,
            A_np=A_np,
            n_mc=dgp_benchmark_n_mc,
            seed_base=dgp_benchmark_seed_base,
            n_workers=int(args.n_workers),
            threads_per_worker=int(args.threads_per_worker),
            report_path="MASTER_SUMMARY_REPORT.txt",
            tracked_dir=dgp_benchmark_tracked_dir,
            untracked_dir=dgp_benchmark_untracked_dir,
            dgp_description=(
                f"Structured REAL_DGP estimator benchmark for dgp_label='{dgp_label}', "
                f"n_mc={dgp_benchmark_n_mc}, seed_base={dgp_benchmark_seed_base}, "
                f"true_ate={float(dgp_truth['ate_true']):.6f}"
            ),
            project_root=str(PROJECT_ROOT),
            report_context=report_context,
            save_csv=True,
            write_report=True,
        )
        dgp_benchmark_source = "computed_and_saved" if use_shared_dgp_cache else "computed_job_local"
    elif dgp_perf_df is not None:
        dgp_benchmark_source = "loaded"
        if dgp_benchmark_report is not None and not dgp_benchmark_report.exists():
            dgp_benchmark_tracked_dir.mkdir(parents=True, exist_ok=True)
            write_master_report(
                out_path=dgp_benchmark_report,
                perf_df=dgp_perf_df,
                dist_df=dgp_dist_df,
                dgp_description=(
                    f"Structured REAL_DGP estimator benchmark for dgp_label='{dgp_label}', "
                    f"n_mc={dgp_benchmark_n_mc}, seed_base={dgp_benchmark_seed_base}, "
                    f"true_ate={float(dgp_truth['ate_true']):.6f}"
                ),
                project_root=str(PROJECT_ROOT),
                execution_summary=report_context,
            )

    print(f"  Source:        {dgp_benchmark_source}")
    if dgp_perf_df is None:
        print("  Report path:   (skipped; use the archived per-seed REAL_DGP benchmark when merging)")
        print("  Summary CSV:   (skipped)")
    else:
        if dgp_benchmark_report is not None:
            print(f"  Report path:   {dgp_benchmark_report}")
        else:
            print("  Report path:   (disabled)")
        print(f"  Summary CSV:   {dgp_benchmark_summary_csv}")
        print_ranked_generator_table(
            dgp_perf_df,
            generator="REAL_DGP",
            heading="REAL_DGP estimator rankings (lower mc_mse is better)",
        )

    generators = {}
    true_ate_by_generator = {}
    dgp_true_ate = float(dgp_truth["ate_true"])

    if "OLS" in requested_generators:
        print("\n[OLS generator] fitting quadratic mean and variance models...")
        t0 = time.time()
        ols_bundle = fit_ols_mean_and_variance_poly(
            X_df=X_df,
            A_df=A_np,
            Y_df=Y_np,
            **ols_shared_params,
        )
        mu0_ols, mu1_ols, sigma0_ols, sigma1_ols = cache_ols_mu_sigma(ols_bundle, X_df)
        print(f"[OLS generator] fit finished in {time.time() - t0:.2f}s")

        print("\n[OLS generator] running diagnostics...")
        t0 = time.time()
        run_cached_generator_diagnostics(
            out_prefix="ols_mean_var",
            generator_label="OLS",
            X_df=X_df,
            A_np=A_np,
            Y_np=Y_np,
            mu0=mu0_ols,
            mu1=mu1_ols,
            sigma0=sigma0_ols,
            sigma1=sigma1_ols,
            out_tracked=out_tracked,
            out_untracked=out_untracked,
            noise=noise,
            t_df=t_df,
            seed_fit=int(seed_plan["ols_diag_fit"]),
            seed_sample=int(seed_plan["ols_diag_sample"]),
            n_samples_mc=10000,
            bivar_cols=bivar_cols,
            write_tracked_artifacts=full_output,
        )
        print(f"[OLS generator] diagnostics finished in {time.time() - t0:.2f}s")

        ols_gen = OLSGenerator(
            mu0=mu0_ols, mu1=mu1_ols, sigma0=sigma0_ols, sigma1=sigma1_ols,
            A_vec=A_np, noise_type=noise, t_df=t_df, clip_sigma_min=0.0,
        )
        generators["OLS"] = ols_gen
        true_ate_by_generator["OLS"] = float(np.mean(ols_gen.mu1 - ols_gen.mu0))

    if "GB" in requested_generators:
        print("\n[GB generator] fitting mean and variance models...")
        t0 = time.time()
        mean_model_gb, var_model_gb = fit_gb_mean_and_variance_cv(
            X_df=X_df,
            A_df=A_np,
            Y_df=Y_np,
            seed=int(seed_plan["gb_fit"]),
            n_splits=5,
            max_iter_mean=gb_shared_params["max_iter"],
            learning_rate_mean=gb_shared_params["learning_rate"],
            max_depth_mean=gb_shared_params["max_depth"],
            max_iter_var=gb_shared_params["max_iter"],
            learning_rate_var=gb_shared_params["learning_rate"],
            max_depth_var=gb_shared_params["max_depth"],
        )
        mu0_gb, mu1_gb, sigma0_gb, sigma1_gb = cache_gb_mu_sigma(mean_model_gb, var_model_gb, X_df)
        print(f"[GB generator] fit finished in {time.time() - t0:.2f}s")

        print("\n[GB generator] running diagnostics...")
        t0 = time.time()
        run_cached_generator_diagnostics(
            out_prefix="gb_mean_var",
            generator_label="GB",
            X_df=X_df,
            A_np=A_np,
            Y_np=Y_np,
            mu0=mu0_gb,
            mu1=mu1_gb,
            sigma0=sigma0_gb,
            sigma1=sigma1_gb,
            out_tracked=out_tracked,
            out_untracked=out_untracked,
            noise=noise,
            t_df=t_df,
            seed_fit=int(seed_plan["gb_diag_fit"]),
            seed_sample=int(seed_plan["gb_diag_sample"]),
            n_samples_mc=10000,
            bivar_cols=bivar_cols,
            write_tracked_artifacts=full_output,
        )
        print(f"[GB generator] diagnostics finished in {time.time() - t0:.2f}s")

        gb_gen = GBGenerator(
            mu0=mu0_gb, mu1=mu1_gb, sigma0=sigma0_gb, sigma1=sigma1_gb,
            A_vec=A_np, noise_type=noise, t_df=t_df, clip_sigma_min=0.0,
        )
        generators["GB"] = gb_gen
        true_ate_by_generator["GB"] = float(np.mean(gb_gen.mu1 - gb_gen.mu0))

    if "RF" in requested_generators:
        print("\n[RF generator] fitting mean and variance models...")
        t0 = time.time()
        mean_model_rf, var_model_rf = fit_rf_mean_and_variance_oob(
            X_df=X_df,
            A_df=A_np,
            Y_df=Y_np,
            seed=int(seed_plan["rf_fit"]),
            **rf_shared_params,
        )
        mu0_rf, mu1_rf, sigma0_rf, sigma1_rf = cache_rf_mu_sigma(mean_model_rf, var_model_rf, X_df)
        print(f"[RF generator] fit finished in {time.time() - t0:.2f}s")

        print("\n[RF generator] running diagnostics...")
        t0 = time.time()
        run_cached_generator_diagnostics(
            out_prefix="rf_mean_var",
            generator_label="RF",
            X_df=X_df,
            A_np=A_np,
            Y_np=Y_np,
            mu0=mu0_rf,
            mu1=mu1_rf,
            sigma0=sigma0_rf,
            sigma1=sigma1_rf,
            out_tracked=out_tracked,
            out_untracked=out_untracked,
            noise=noise,
            t_df=t_df,
            seed_fit=int(seed_plan["rf_diag_fit"]),
            seed_sample=int(seed_plan["rf_diag_sample"]),
            n_samples_mc=10000,
            bivar_cols=bivar_cols,
            write_tracked_artifacts=full_output,
        )
        print(f"[RF generator] diagnostics finished in {time.time() - t0:.2f}s")

        rf_gen = RFGenerator(
            mu0=mu0_rf, mu1=mu1_rf, sigma0=sigma0_rf, sigma1=sigma1_rf,
            A_vec=A_np, noise_type=noise, t_df=t_df, clip_sigma_min=0.0,
        )
        generators["RF"] = rf_gen
        true_ate_by_generator["RF"] = float(np.mean(rf_gen.mu1 - rf_gen.mu0))

    if "NN" in requested_generators:
        print("\n[NN generator] fitting mean and variance models...")
        t0 = time.time()
        mean_model_nn, var_model_nn = fit_nn_mean_and_variance_cv(
            X_df=X_df,
            A_df=A_np,
            Y_df=Y_np,
            seed=int(seed_plan["nn_fit"]),
            n_splits=5,
            **nn_shared_params,
        )
        mu0_nn, mu1_nn, sigma0_nn, sigma1_nn = cache_nn_mu_sigma(mean_model_nn, var_model_nn, X_df)
        print(f"[NN generator] fit finished in {time.time() - t0:.2f}s")

        print("\n[NN generator] running diagnostics...")
        t0 = time.time()
        run_cached_generator_diagnostics(
            out_prefix="nn_mean_var",
            generator_label="NN",
            X_df=X_df,
            A_np=A_np,
            Y_np=Y_np,
            mu0=mu0_nn,
            mu1=mu1_nn,
            sigma0=sigma0_nn,
            sigma1=sigma1_nn,
            out_tracked=out_tracked,
            out_untracked=out_untracked,
            noise=noise,
            t_df=t_df,
            seed_fit=int(seed_plan["nn_diag_fit"]),
            seed_sample=int(seed_plan["nn_diag_sample"]),
            n_samples_mc=10000,
            bivar_cols=bivar_cols,
            write_tracked_artifacts=full_output,
        )
        print(f"[NN generator] diagnostics finished in {time.time() - t0:.2f}s")

        nn_gen = NNGenerator(
            mu0=mu0_nn, mu1=mu1_nn, sigma0=sigma0_nn, sigma1=sigma1_nn,
            A_vec=A_np, noise_type=noise, t_df=t_df, clip_sigma_min=0.0,
        )
        generators["NN"] = nn_gen
        true_ate_by_generator["NN"] = float(np.mean(nn_gen.mu1 - nn_gen.mu0))


    if "WGAN" in requested_generators:
        from wgan_module import WGANmodel

        print("\n[WGAN generator] initializing model...")
        wgan = WGANmodel(
            covariate_dim=X_df.shape[1],
            noise_dim=10,
            hidden_dim=128,
            n_layers=3,
            lambda_gp=10.0,
            n_critic=5,
            g_lr=5e-5,
            c_lr=2e-4,
            target_ate=None,
            lambda_ate=1.0,
            initialization_seed=int(seed_plan["wgan_init"]),
        )
        print("[WGAN generator] fitting...")
        t0 = time.time()
        wgan.fit(
            X=X_df.to_numpy(dtype="float32"),
            A=A_np.reshape(-1).astype("float32"),
            Y=Y_np.astype("float32"),
            batch_size=int(args.wgan_batch_size),
            n_epochs=int(args.wgan_epochs),
            print_every=max(1, int(args.wgan_print_every)),
            seed=int(seed_plan["wgan_fit"]),
        )
        wgan_gen = WGANGenerator(
            fitted_wgan=wgan,
            X=X_df.to_numpy(dtype="float32"),
            A_vec=A_np,
        )
        print(f"[WGAN generator] fit finished in {time.time() - t0:.2f}s")

        print("\n[WGAN generator] running diagnostics...")
        t0 = time.time()
        wgan_diag = run_wgan_generator_diagnostics(
            out_prefix="wgan",
            generator_label="WGAN",
            fitted_wgan=wgan,
            X_df=X_df,
            A_np=A_np,
            Y_np=Y_np,
            out_tracked=out_tracked,
            out_untracked=out_untracked,
            seed_fit=int(seed_plan["wgan_diag_fit"]),
            seed_sample=int(seed_plan["wgan_diag_sample"]),
            n_samples_mc=10000,
            bivar_cols=bivar_cols,
            write_tracked_artifacts=full_output,
        )
        print(f"[WGAN generator] diagnostics finished in {time.time() - t0:.2f}s")

        generators["WGAN"] = wgan_gen
        true_ate_by_generator["WGAN"] = float(wgan_diag["summary"]["true_ate_mc"])

    generators = {name: generators[name] for name in requested_generators if name in generators}
    true_ate_by_generator = {
        name: true_ate_by_generator[name]
        for name in requested_generators
        if name in true_ate_by_generator
    }
    print("Generator diagnostics saved" if full_output else "Generator diagnostics CSVs saved")

    tracked_run_dir, untracked_run_dir = resolve_run_output_dirs(
        out_root=out_root,
        dgp_label=dgp_label,
        run_label=run_label,
        run_id=args.run_id,
        full_output=full_output,
    )
    if tracked_run_dir is not None:
        tracked_run_dir.mkdir(parents=True, exist_ok=True)
    untracked_run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("START OF THE MONTE CARLO STUDY")
    print("=" * 80)
    print("Generators selected:")
    for gen_name in generators:
        print(f"  - {gen_name}")
    print("Estimators selected:")
    for est_name in estimators:
        print(f"  - {est_name}")
    print("=" * 80)

    perf_df, dist_df = run_mc_study(
        generators=generators,
        y_ref=Y_np,
        true_ate_by_generator=true_ate_by_generator,
        estimators=estimators,
        estimator_params=estimator_params,
        X_df=X_df,
        A_np=A_np,
        n_mc=int(args.n_mc),
        seed_base=int(seed_plan["mc_base"]),
        n_workers=int(args.n_workers),
        threads_per_worker=int(args.threads_per_worker),
        report_path="MASTER_SUMMARY_REPORT.txt",
        tracked_dir=tracked_run_dir,
        untracked_dir=untracked_run_dir,
        dgp_description=(
            f"Observed-data synthetic generator comparison using outcome='{outcome_label}', "
            f"treatment='{args.treatment_col}', run_label='{run_label if run_label else '(none)'}', "
            f"job_seed={seed_plan['job_seed']}, mc_base={seed_plan['mc_base']}, "
            f"dgp_label='{dgp_label}', dgp_truth_path='{dgp_truth_path}', "
            f"dgp_true_ate={dgp_true_ate:.6f}"
        ),
        real_dgp_perf_df=dgp_perf_df,
        project_root=str(PROJECT_ROOT),
        report_context=report_context,
        save_csv=True,
        write_report=True,
    )

    print(perf_df.head(10))
    print(dist_df.head(10))
    print(f"\nFull study finished in {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
