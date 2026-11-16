"""Causal estimators and shared cross-fitted nuisance-model utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def preload_estimator_dependencies(estimator_names: list[str]) -> None:
    """Check optional dependencies before starting Monte Carlo workers."""
    from importlib import import_module
    import warnings

    names = set(estimator_names)
    if names.intersection({"OLS1", "OLS2", "OLS2_GCOMP"}):
        import_module("statsmodels.api")
    if "BCF_GCOMP" in names:
        warnings.filterwarnings("ignore", category=UserWarning, module=r"stochtree\.bart")
        getattr(import_module("stochtree"), "BCFModel")

# ------------------------------------------------------------------------------
# ORACLE OLS (unchanged)
# ------------------------------------------------------------------------------
def oracle_ols(
    X_df: pd.DataFrame,
    A_np,
    y,
    *,
    coef_dict: dict,
    true_ate=None,
    seed: int = 0,
    alpha: float = 0.05,
    n_boot=None,
):
    import statsmodels.api as sm

    y = np.asarray(y).reshape(-1)
    A = np.asarray(A_np).reshape(-1).astype(int)

    compas  = X_df["compas"].astype(float).to_numpy()
    career  = X_df["career_ladder"].astype(float).to_numpy()
    diploma = X_df["diploma"].astype(float).to_numpy()
    age     = X_df["age"].astype(float).to_numpy()

    age_c = age - age.mean()
    age3  = age_c ** 3

    job_cols     = [c for c in X_df.columns if c.startswith("job_domain_")]
    comporg_cols = [c for c in X_df.columns if c.startswith("company_Organization.")]

    Xo = pd.DataFrame(index=X_df.index)
    Xo["age_c"] = age_c
    Xo["age3"] = age3
    Xo["compas"] = compas
    Xo["career_ladder"] = career
    Xo["diploma"] = diploma
    if job_cols:
        Xo[job_cols] = X_df[job_cols].astype(float)
    if comporg_cols:
        Xo[comporg_cols] = X_df[comporg_cols].astype(float)

    exp_lin = (
        float(coef_dict["C_exp"])
        + float(coef_dict["b_age_exp"]) * age_c
        + float(coef_dict["b_compas_exp"]) * compas
    )
    Xo["exp_term"] = np.exp(exp_lin)

    X_all = Xo.copy()
    X_all.insert(0, "A", A)
    X_all = sm.add_constant(X_all, has_constant="add")

    fit = sm.OLS(y, X_all).fit()

    ate_hat = float(fit.params["A"])
    se_hat  = float(fit.bse["A"])

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat)
    ci_hi = float(ate_hat + z * se_hat)
    cover = float((true_ate is not None) and (true_ate >= ci_lo) and (true_ate <= ci_hi))
    return ate_hat, se_hat, ci_lo, ci_hi, cover


# ------------------------------------------------------------------------------
# OLS (unchanged)
# ------------------------------------------------------------------------------
def ols(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    cov_type="HC3",
):
    import statsmodels.api as sm

    y = np.asarray(y).reshape(-1)
    A = np.asarray(A_np).reshape(-1)

    X_all = X_df.copy()
    X_all["A"] = A
    X_all = sm.add_constant(X_all, has_constant="add")

    fit = sm.OLS(y, X_all).fit(cov_type=cov_type)

    ate_hat = float(fit.params["A"])
    se_hat  = float(fit.bse["A"])

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat)
    ci_hi = float(ate_hat + z * se_hat)
    cover = float((true_ate is not None) and (true_ate >= ci_lo) and (true_ate <= ci_hi))
    return ate_hat, se_hat, ci_lo, ci_hi, cover


def ols_poly_gcomp(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    degree: int = 2,
    include_bias: bool = False,
    shared_outcome_fit=None,
    se_method: str = "plugin",   # "plugin" | "bayes_boot" | "none"
    n_boot: int = 300,
):
    rng = np.random.default_rng(int(seed))

    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X_df = X_df.reset_index(drop=True)
    n = len(X_df)
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="ols2", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        X_all, _, _ = _build_outcome_designs(X_df, A)
        model = _PolynomialOLSOutcomeRegressor(
            degree=int(degree),
            include_bias=bool(include_bias),
        )
        model.fit(np.asarray(X_all, dtype=float), y)
        mu0, mu1 = _predict_counterfactuals_from_shared_model(model, X_df)

    d = mu1 - mu0
    ate_hat = float(d.mean())

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan

    z = float(norm.ppf(1 - alpha / 2))

    if se_method == "plugin":
        se_hat = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    elif se_method == "bayes_boot":
        B = int(n_boot)
        ate_b = np.empty(B, dtype=float)
        for b in range(B):
            w = rng.exponential(scale=1.0, size=n)
            ate_b[b] = float(np.sum(w * d) / np.sum(w))
        se_hat = float(ate_b.std(ddof=1)) if B > 1 else float("nan")
    else:
        raise ValueError("se_method must be 'plugin', 'bayes_boot', or 'none'")

    ci_lo = float(ate_hat - z * se_hat) if np.isfinite(se_hat) else float("nan")
    ci_hi = float(ate_hat + z * se_hat) if np.isfinite(se_hat) else float("nan")
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover


# ==============================================================================
# GB_IPW (unchanged; supports fixed PS)
# ==============================================================================
def gb_ipw(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    stabilized=True,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X = X_df.to_numpy(dtype=float)
    n = X.shape[0]
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    p1, _ = _resolve_propensity_inputs(
        X=X,
        n=n,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        seed=seed,
        A=A,
        allow_refit=True,
    )
    p0 = 1.0 - p1

    if stabilized:
        pi1 = A.mean()
        pi0 = 1.0 - pi1
        w1 = (A * pi1) / p1
        w0 = ((1 - A) * pi0) / p0
    else:
        w1 = A / p1
        w0 = (1 - A) / p0

    sw1 = w1.sum()
    sw0 = w0.sum()
    if sw1 <= 0 or sw0 <= 0:
        raise ValueError(f"Degenerate weights: sum(w1)={sw1}, sum(w0)={sw0}")

    mu1 = (w1 * y).sum() / sw1
    mu0 = (w0 * y).sum() / sw0
    ate_hat = float(mu1 - mu0)

    var1 = float((w1**2 * (y - mu1)**2).sum() / (sw1**2))
    var0 = float((w0**2 * (y - mu0)**2).sum() / (sw0**2))
    se_hat = float(np.sqrt(var1 + var0))

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat)
    ci_hi = float(ate_hat + z * se_hat)
    cover = float((true_ate is not None) and (true_ate >= ci_lo) and (true_ate <= ci_hi))
    return ate_hat, se_hat, ci_lo, ci_hi, cover


# ==============================================================================
# Shared outcome nuisance helpers
# ==============================================================================
_SHARED_OUTCOME_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "OLS2_GCOMP": ("ols2", ("degree", "include_bias", "crossfit_n_splits", "crossfit_stratified")),
    "GB_OLS2_AIPW": ("ols2", ("degree", "include_bias", "crossfit_n_splits", "crossfit_stratified")),
    "GB_OLS2_TMLE": ("ols2", ("degree", "include_bias", "crossfit_n_splits", "crossfit_stratified")),
    "GB_GCOMP": ("gb", ("max_iter", "learning_rate", "max_depth", "crossfit_n_splits", "crossfit_stratified")),
    "GB_GB_AIPW": ("gb", ("max_iter", "learning_rate", "max_depth", "crossfit_n_splits", "crossfit_stratified")),
    "GB_GB_TMLE": ("gb", ("max_iter", "learning_rate", "max_depth", "crossfit_n_splits", "crossfit_stratified")),
    "RF_GCOMP": ("rf", ("rf_ntrees", "rf_minleaf", "rf_maxfeat", "rf_bootstrap", "rf_max_depth", "crossfit_n_splits", "crossfit_stratified")),
    "GB_RF_AIPW": ("rf", ("rf_ntrees", "rf_minleaf", "rf_maxfeat", "rf_bootstrap", "rf_max_depth", "crossfit_n_splits", "crossfit_stratified")),
    "GB_RF_TMLE": ("rf", ("rf_ntrees", "rf_minleaf", "rf_maxfeat", "rf_bootstrap", "rf_max_depth", "crossfit_n_splits", "crossfit_stratified")),
    "NN_GCOMP": (
        "nn",
        (
            "hidden_dims",
            "lr",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
            "min_delta",
            "crossfit_n_splits",
            "crossfit_stratified",
        ),
    ),
    "GB_NN_AIPW": (
        "nn",
        (
            "hidden_dims",
            "lr",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
            "min_delta",
            "crossfit_n_splits",
            "crossfit_stratified",
        ),
    ),
    "GB_NN_TMLE": (
        "nn",
        (
            "hidden_dims",
            "lr",
            "weight_decay",
            "batch_size",
            "max_epochs",
            "patience",
            "min_delta",
            "crossfit_n_splits",
            "crossfit_stratified",
        ),
    ),
    "BCF_GCOMP": (
        "bcf",
        (
            "num_gfr",
            "num_burnin",
            "num_mcmc",
            "prognostic_trees",
            "treatment_trees",
            "prognostic_alpha",
            "prognostic_beta",
            "treatment_alpha",
            "treatment_beta",
            "standardize",
            "num_chains",
            "heteroskedastic",
            "var_trees",
            "min_propensity",
        ),
    ),
}

_SHARED_PROPENSITY_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "GB_IPW": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_GB_AIPW": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_RF_AIPW": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_OLS2_AIPW": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_NN_AIPW": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_GB_TMLE": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_RF_TMLE": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_OLS2_TMLE": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
    "GB_NN_TMLE": ("gb", ("crossfit_n_splits", "crossfit_stratified")),
}


def _freeze_shared_param(value):
    if isinstance(value, np.ndarray):
        return tuple(_freeze_shared_param(v) for v in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_shared_param(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_shared_param(v)) for k, v in value.items()))
    return value


def shared_outcome_fit_key(estimator_label: str, params: dict | None = None):
    spec = _SHARED_OUTCOME_SPECS.get(str(estimator_label))
    if spec is None:
        return None
    family, param_names = spec
    params = params or {}
    key_values = tuple((name, _freeze_shared_param(params.get(name))) for name in param_names)
    return family, key_values


def shared_propensity_fit_key(estimator_label: str, params: dict | None = None):
    spec = _SHARED_PROPENSITY_SPECS.get(str(estimator_label))
    if spec is None:
        return None
    family, param_names = spec
    params = params or {}
    key_values = tuple((name, _freeze_shared_param(params.get(name))) for name in param_names)
    return family, key_values


def _build_outcome_designs(X_df, A_np):
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X_base = X_df.reset_index(drop=True).copy()
    X_all = X_base.copy()
    X_all["A"] = A
    X0 = X_base.copy()
    X0["A"] = 0
    X1 = X_base.copy()
    X1["A"] = 1
    return X_all, X0, X1


def _crossfit_splits(
    *,
    n: int,
    A,
    n_splits: int,
    seed: int,
    stratified: bool,
):
    n_splits = int(n_splits)
    if n_splits <= 1:
        return None
    if n_splits > n:
        raise ValueError(f"crossfit_n_splits={n_splits} exceeds sample size n={n}.")

    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        return list(splitter.split(np.zeros(n), np.asarray(A, dtype=int).reshape(-1)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return list(splitter.split(np.arange(n)))


def _predict_counterfactuals_from_shared_model(model, X_df):
    _, X0, X1 = _build_outcome_designs(X_df, np.zeros(len(X_df), dtype=int))
    mu0 = np.asarray(model.predict(np.asarray(X0, dtype=float)), dtype=float).reshape(-1)
    mu1 = np.asarray(model.predict(np.asarray(X1, dtype=float)), dtype=float).reshape(-1)
    return mu0, mu1


def _validate_shared_outcome_fit(shared_outcome_fit, *, family: str, n: int):
    if shared_outcome_fit is None:
        return None
    if str(shared_outcome_fit.get("family")) != str(family):
        raise ValueError(
            f"Shared outcome fit family mismatch: expected '{family}' but found "
            f"'{shared_outcome_fit.get('family')}'."
        )
    mu0 = np.asarray(shared_outcome_fit["mu0"], dtype=float).reshape(-1)
    mu1 = np.asarray(shared_outcome_fit["mu1"], dtype=float).reshape(-1)
    if mu0.shape[0] != n or mu1.shape[0] != n:
        raise ValueError(
            f"Shared outcome fit length mismatch: expected {n}, got "
            f"mu0={mu0.shape[0]} and mu1={mu1.shape[0]}."
        )
    return shared_outcome_fit


def _validate_shared_propensity_fit(shared_propensity_fit, *, family: str, n: int):
    if shared_propensity_fit is None:
        return None
    if str(shared_propensity_fit.get("family")) != str(family):
        raise ValueError(
            f"Shared propensity fit family mismatch: expected '{family}' but found "
            f"'{shared_propensity_fit.get('family')}'."
        )
    p1 = np.asarray(shared_propensity_fit["p1"], dtype=float).reshape(-1)
    if p1.shape[0] != n:
        raise ValueError(
            f"Shared propensity fit length mismatch: expected {n}, got p1={p1.shape[0]}."
        )
    return shared_propensity_fit


def fit_shared_propensity_for_estimator(
    estimator_label: str,
    X_df,
    A_np,
    *,
    seed: int = 0,
    **params,
):
    spec = _SHARED_PROPENSITY_SPECS.get(str(estimator_label))
    if spec is None:
        raise ValueError(f"Estimator '{estimator_label}' does not use a shareable propensity nuisance model.")

    family, _ = spec
    X = np.asarray(X_df, dtype=float)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    n = X.shape[0]
    crossfit_n_splits = int(params.get("crossfit_n_splits", 1))
    crossfit_stratified = bool(params.get("crossfit_stratified", True))

    if family == "gb":
        splits = _crossfit_splits(
            n=n,
            A=A,
            n_splits=crossfit_n_splits,
            seed=int(seed),
            stratified=crossfit_stratified,
        )
        if splits is None:
            model = GradientBoostingClassifier(random_state=int(seed))
            model.fit(X, A)
            p1 = np.asarray(model.predict_proba(X)[:, 1], dtype=float).reshape(-1)
        else:
            p1 = np.empty(n, dtype=float)
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                fold_model = GradientBoostingClassifier(random_state=int(seed) + fold_id)
                fold_model.fit(X[train_idx], A[train_idx])
                p1[test_idx] = np.asarray(
                    fold_model.predict_proba(X[test_idx])[:, 1],
                    dtype=float,
                ).reshape(-1)
            model = None
        return {
            "family": family,
            "model": model,
            "p1": p1,
        }
    raise ValueError(f"Unsupported shared propensity family '{family}'.")


def _resolve_propensity_inputs(
    *,
    X,
    n: int,
    trim: float,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    seed: int = 0,
    A=None,
    allow_refit: bool = False,
):
    exp_model = None
    shared = _validate_shared_propensity_fit(shared_propensity_fit, family="gb", n=n)
    if shared is not None:
        p1 = np.asarray(shared["p1"], dtype=float).reshape(-1)
        exp_model = _FixedPropensityModel(p1_fixed=p1, trim=trim)
    elif p1_fixed is not None:
        p1 = np.asarray(p1_fixed, dtype=float).reshape(-1)
        if p1.shape[0] != n:
            raise ValueError(f"p1_fixed has length {p1.shape[0]} but X has {n} rows.")
        exp_model = _FixedPropensityModel(p1_fixed=p1, trim=trim)
    elif ps_model is not None:
        p1 = np.asarray(ps_model.predict_proba(X)[:, 1], dtype=float).reshape(-1)
        exp_model = ps_model
    elif allow_refit:
        if A is None:
            raise ValueError("A must be provided when allow_refit=True.")
        ps = GradientBoostingClassifier(random_state=int(seed))
        ps.fit(X, np.asarray(A, dtype=int).reshape(-1))
        p1 = np.asarray(ps.predict_proba(X)[:, 1], dtype=float).reshape(-1)
        exp_model = ps
    else:
        raise ValueError("Provide shared_propensity_fit, p1_fixed, or prefit ps_model.")
    p1 = np.clip(p1, trim, 1.0 - trim)
    return p1, exp_model


def fit_shared_outcome_for_estimator(
    estimator_label: str,
    X_df,
    A_np,
    y,
    *,
    seed: int = 0,
    **params,
):
    spec = _SHARED_OUTCOME_SPECS.get(str(estimator_label))
    if spec is None:
        raise ValueError(f"Estimator '{estimator_label}' does not use a shareable outcome nuisance model.")

    family, _ = spec
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X_all, _, _ = _build_outcome_designs(X_df, A)
    X_all_np = np.asarray(X_all, dtype=float)
    n = X_all_np.shape[0]
    crossfit_n_splits = int(params.get("crossfit_n_splits", 1))
    crossfit_stratified = bool(params.get("crossfit_stratified", True))
    need_full_model = bool(
        family == "rf" and str(params.get("se_method", "")).lower() == "subforest"
    )

    def _make_point_model(local_seed: int):
        if family == "ols2":
            return _PolynomialOLSOutcomeRegressor(
                degree=int(params.get("degree", 2)),
                include_bias=bool(params.get("include_bias", False)),
            )
        if family == "gb":
            return _HistGBOutcomeRegressor(
                max_iter=int(params.get("max_iter", 500)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                max_depth=params.get("max_depth", 6),
                seed=int(local_seed),
            )
        if family == "rf":
            return _RFOutcomeRegressor(
                rf_ntrees=int(params.get("rf_ntrees", 800)),
                rf_minleaf=int(params.get("rf_minleaf", 1)),
                rf_maxfeat=params.get("rf_maxfeat", 1.0),
                rf_bootstrap=bool(params.get("rf_bootstrap", True)),
                rf_max_depth=params.get("rf_max_depth", None),
                rf_n_jobs=int(params.get("rf_n_jobs", 1)),
                seed=int(local_seed),
            )
        if family == "nn":
            return _PooledNNOutcomeRegressor(
                hidden_dims=tuple(params.get("hidden_dims", (64, 32))),
                lr=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 1e-4)),
                batch_size=int(params.get("batch_size", 256)),
                max_epochs=int(params.get("max_epochs", 400)),
                patience=int(params.get("patience", 25)),
                min_delta=float(params.get("min_delta", 1e-3)),
                seed=int(local_seed),
            )
        raise ValueError(f"Unsupported cross-fitted outcome family '{family}'.")

    if family in {"ols2", "gb", "rf", "nn"}:
        splits = _crossfit_splits(
            n=n,
            A=A,
            n_splits=crossfit_n_splits,
            seed=int(seed),
            stratified=crossfit_stratified,
        )
        if splits is None:
            model = _make_point_model(int(seed))
            model.fit(X_all_np, y)
            mu0, mu1 = _predict_counterfactuals_from_shared_model(model, X_df)
        else:
            mu0 = np.empty(n, dtype=float)
            mu1 = np.empty(n, dtype=float)
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                fold_model = _make_point_model(int(seed) + fold_id)
                fold_model.fit(X_all_np[train_idx], y[train_idx])
                X_hold = X_df.iloc[test_idx].reset_index(drop=True)
                mu0_fold, mu1_fold = _predict_counterfactuals_from_shared_model(fold_model, X_hold)
                mu0[test_idx] = np.asarray(mu0_fold, dtype=float).reshape(-1)
                mu1[test_idx] = np.asarray(mu1_fold, dtype=float).reshape(-1)
            if need_full_model:
                model = _make_point_model(int(seed))
                model.fit(X_all_np, y)
            else:
                model = None
        return {
            "family": family,
            "model": model,
            "mu0": mu0,
            "mu1": mu1,
        }

    if family == "bcf":
        return _fit_shared_bcf_outcome(
            X_df,
            A,
            y,
            seed=int(seed),
            p1_fixed=params.get("p1_fixed"),
            num_gfr=int(params.get("num_gfr", 10)),
            num_burnin=int(params.get("num_burnin", 20)),
            num_mcmc=int(params.get("num_mcmc", 30)),
            prognostic_trees=int(params.get("prognostic_trees", 250)),
            treatment_trees=int(params.get("treatment_trees", 50)),
            prognostic_alpha=float(params.get("prognostic_alpha", 0.95)),
            prognostic_beta=float(params.get("prognostic_beta", 2.0)),
            treatment_alpha=float(params.get("treatment_alpha", 0.25)),
            treatment_beta=float(params.get("treatment_beta", 3.0)),
            standardize=bool(params.get("standardize", True)),
            num_threads=int(params.get("num_threads", 0)),
            num_chains=int(params.get("num_chains", 1)),
            heteroskedastic=bool(params.get("heteroskedastic", False)),
            var_trees=int(params.get("var_trees", 20)),
            min_propensity=float(params.get("min_propensity", 1e-3)),
        )
    else:
        raise ValueError(f"Unsupported shared outcome family '{family}'.")

    model.fit(X_all_np, y)
    mu0, mu1 = _predict_counterfactuals_from_shared_model(model, X_df)
    return {
        "family": family,
        "model": model,
        "mu0": mu0,
        "mu1": mu1,
    }


# ==============================================================================
# GB_GCOMP (ONE HistGB on (X,A)) — SE OPTIONAL
#   Hyperparams MUST be set to the same values you used to fit/cache the GB mean model.
# ==============================================================================
def gb_gcomp(
    X_df: pd.DataFrame,
    A_np,
    y,
    *,
    true_ate=None,
    seed: int = 42,
    alpha: float = 0.05,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    shared_outcome_fit=None,
    se_method: str = "plugin",   # "plugin" | "bayes_boot" | "none"
    n_boot: int = 300,
):
    rng = np.random.default_rng(int(seed))

    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    n = len(X_df)
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="gb", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        X_all = X_df.reset_index(drop=True).copy()
        X_all["A"] = A

        m = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            max_depth=max_depth,
            random_state=int(seed),
        )
        m.fit(X_all, y)

        X0 = X_df.reset_index(drop=True).copy(); X0["A"] = 0
        X1 = X_df.reset_index(drop=True).copy(); X1["A"] = 1

        mu0 = m.predict(X0).astype(float).reshape(-1)
        mu1 = m.predict(X1).astype(float).reshape(-1)

    d = mu1 - mu0
    ate_hat = float(d.mean())

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan

    z = float(norm.ppf(1 - alpha / 2))

    if se_method == "plugin":
        se_hat = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    elif se_method == "bayes_boot":
        B = int(n_boot)
        ate_b = np.empty(B, dtype=float)
        for b in range(B):
            w = rng.exponential(scale=1.0, size=n)
            ate_b[b] = float(np.sum(w * d) / np.sum(w))
        se_hat = float(ate_b.std(ddof=1)) if B > 1 else float("nan")
    else:
        raise ValueError("se_method must be 'plugin', 'bayes_boot', or 'none'")

    ci_lo = float(ate_hat - z * se_hat) if np.isfinite(se_hat) else float("nan")
    ci_hi = float(ate_hat + z * se_hat) if np.isfinite(se_hat) else float("nan")
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover


# ==============================================================================
# RF_GCOMP (ONE RF on (X,A)) — SE OPTIONAL
#   Hyperparams MUST be set to same values you used to fit/cache the RF mean model.
# ==============================================================================
def rf_gcomp(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    rf_ntrees=800,
    rf_minleaf=1,
    rf_maxfeat=1.0,
    rf_bootstrap=True,
    rf_max_depth=None,
    rf_n_jobs=1,
    shared_outcome_fit=None,
    se_method="subforest",   # "subforest" | "none"
    K=10,
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X_df = X_df.reset_index(drop=True)
    n = len(X_df)
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    X0 = X_df.copy(); X0["A"] = 0
    X1 = X_df.copy(); X1["A"] = 1
    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="rf", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
        rf = getattr(shared.get("model"), "model_", None)
        if rf is None and se_method == "subforest":
            raise ValueError("Shared RF outcome fit is missing the fitted forest needed for subforest SEs.")
    else:
        X_all = X_df.copy()
        X_all["A"] = A

        rf = RandomForestRegressor(
            n_estimators=int(rf_ntrees),
            min_samples_leaf=int(rf_minleaf),
            max_features=rf_maxfeat,
            bootstrap=bool(rf_bootstrap),
            max_depth=rf_max_depth,
            n_jobs=int(rf_n_jobs),
            random_state=int(seed),
        )
        rf.fit(X_all, y)

        mu0 = rf.predict(X0).astype(float).reshape(-1)
        mu1 = rf.predict(X1).astype(float).reshape(-1)
    ate_hat = float(np.mean(mu1 - mu0))

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan
    if se_method != "subforest":
        raise ValueError("se_method must be 'subforest' or 'none'")

    # subforest SE
    trees = rf.estimators_
    T = len(trees)
    K_eff = int(min(max(2, int(K)), T))
    idx = np.arange(T)
    rng = np.random.default_rng(int(seed) + 999)
    rng.shuffle(idx)
    blocks = np.array_split(idx, K_eff)

    X0_np = X0.to_numpy(dtype=float)
    X1_np = X1.to_numpy(dtype=float)

    ate_blocks = np.empty(K_eff, dtype=float)
    for k, bidx in enumerate(blocks):
        p0 = np.zeros(n, dtype=float)
        p1 = np.zeros(n, dtype=float)
        for j in bidx:
            p0 += trees[j].predict(X0_np)
            p1 += trees[j].predict(X1_np)
        p0 /= len(bidx)
        p1 /= len(bidx)
        ate_blocks[k] = float(np.mean(p1 - p0))

    se_hat = float(np.std(ate_blocks, ddof=1) / np.sqrt(K_eff))

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat)
    ci_hi = float(ate_hat + z * se_hat)
    cover = float((true_ate is not None) and (true_ate >= ci_lo) and (true_ate <= ci_hi))
    return ate_hat, se_hat, ci_lo, ci_hi, cover


def _fit_pooled_nn_outcome_model(
    X_df,
    A_np,
    y,
    *,
    seed=0,
    hidden_dims=(64, 32),
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=400,
    patience=25,
    min_delta=1e-3,
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X_all, X0, X1 = _build_outcome_designs(X_df, A)

    model = _PooledNNOutcomeRegressor(
        hidden_dims=hidden_dims,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
    )
    model.fit(np.asarray(X_all, dtype=float), y)
    mu0 = np.asarray(model.predict(np.asarray(X0, dtype=float)), dtype=float).reshape(-1)
    mu1 = np.asarray(model.predict(np.asarray(X1, dtype=float)), dtype=float).reshape(-1)
    return mu0, mu1


def nn_gcomp(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    hidden_dims=(64, 32),
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=400,
    patience=25,
    min_delta=1e-3,
    shared_outcome_fit=None,
    se_method="plugin",   # "plugin" | "bayes_boot" | "none"
    n_boot=300,
):
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    n = len(X_df)
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="nn", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        mu0, mu1 = _fit_pooled_nn_outcome_model(
            X_df,
            A,
            y,
            seed=seed,
            hidden_dims=hidden_dims,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )
    d = mu1 - mu0
    ate_hat = float(d.mean())

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan

    z = float(norm.ppf(1 - alpha / 2))
    if se_method == "plugin":
        se_hat = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    elif se_method == "bayes_boot":
        B = int(n_boot)
        ate_b = np.empty(B, dtype=float)
        for b in range(B):
            w = rng.exponential(scale=1.0, size=n)
            ate_b[b] = float(np.sum(w * d) / np.sum(w))
        se_hat = float(ate_b.std(ddof=1)) if B > 1 else float("nan")
    else:
        raise ValueError("se_method must be 'plugin', 'bayes_boot', or 'none'")

    ci_lo = float(ate_hat - z * se_hat) if np.isfinite(se_hat) else float("nan")
    ci_hi = float(ate_hat + z * se_hat) if np.isfinite(se_hat) else float("nan")
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover


def gb_nn_aipw(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    hidden_dims=(64, 32),
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=400,
    patience=25,
    min_delta=1e-3,
    shared_outcome_fit=None,
    se_method="if",   # "if" | "none"
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X = np.asarray(X_df, dtype=float)
    n = X.shape[0]
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    p1, _ = _resolve_propensity_inputs(
        X=X,
        n=n,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        seed=seed,
    )
    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="nn", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        mu0, mu1 = _fit_pooled_nn_outcome_model(
            X_df,
            A,
            y,
            seed=seed,
            hidden_dims=hidden_dims,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )

    return _aipw_from_predictions(
        mu0,
        mu1,
        A,
        y,
        p1,
        true_ate=true_ate,
        alpha=alpha,
        se_method=se_method,
    )


# ==============================================================================
# AIPW MATCHED TO OUTCOME MODEL ARCHITECTURE — SE OPTIONAL
#   - fixed PS via p1_fixed or ps_model (no refit inside)
#   - outcome model is ONE model on (X,A)
# ==============================================================================
def gb_gb_aipw(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    # outcome (GB) hyperparams
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    shared_outcome_fit=None,
    # SE control
    se_method: str = "if",   # "if" | "none"
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X = X_df.to_numpy(dtype=float)
    n = X.shape[0]
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    p1, _ = _resolve_propensity_inputs(
        X=X,
        n=n,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        seed=seed,
    )

    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="gb", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        X_all = X_df.reset_index(drop=True).copy()
        X_all["A"] = A

        m = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(learning_rate),
            max_iter=int(max_iter),
            max_depth=max_depth,
            random_state=int(seed),
        )
        m.fit(X_all, y)

        X0 = X_df.reset_index(drop=True).copy(); X0["A"] = 0
        X1 = X_df.reset_index(drop=True).copy(); X1["A"] = 1
        mu0 = m.predict(X0).astype(float).reshape(-1)
        mu1 = m.predict(X1).astype(float).reshape(-1)

    return _aipw_from_predictions(
        mu0,
        mu1,
        A,
        y,
        p1,
        true_ate=true_ate,
        alpha=alpha,
        se_method=se_method,
    )


def _tmle_safe_term(name: str) -> str:
    escaped = str(name).replace("\\", "\\\\").replace("'", "\\'")
    return f"Q('{escaped}')"


def _tmle_covariate_formula(X_df, exposure_name: str) -> str:
    terms = [exposure_name] + [_tmle_safe_term(c) for c in X_df.columns]
    return " + ".join(terms)


class _FixedPropensityModel(BaseEstimator, ClassifierMixin):
    def __init__(self, p1_fixed, trim: float = 0.01):
        self.p1_fixed = np.asarray(p1_fixed, dtype=float).reshape(-1)
        self.trim = float(trim)

    def fit(self, X, y):
        n = np.asarray(X).shape[0]
        if self.p1_fixed.shape[0] != n:
            raise ValueError(f"p1_fixed has length {self.p1_fixed.shape[0]} but X has {n} rows.")
        self._p1 = np.clip(self.p1_fixed, self.trim, 1.0 - self.trim)
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        if n != self._p1.shape[0]:
            raise ValueError(
                "Fixed propensity model can only be used on the same observation set used during fit()."
            )
        p1 = self._p1.reshape(-1, 1)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])


class _HistGBOutcomeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, max_iter: int = 500, learning_rate: float = 0.05, max_depth: int | None = 6, seed: int = 0):
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.seed = int(seed)

    def fit(self, X, y):
        self.model_ = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            random_state=self.seed,
        )
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1))
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)


class _PolynomialOLSOutcomeRegressor(BaseEstimator, RegressorMixin):
    """Second-order OLS outcome model on covariates and treatment."""

    def __init__(self, degree: int = 2, include_bias: bool = False):
        self.degree = int(degree)
        self.include_bias = bool(include_bias)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
        )
        self.model_ = LinearRegression()
        self.model_.fit(self.poly_.fit_transform(X), y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray(
            self.model_.predict(self.poly_.transform(X)),
            dtype=float,
        ).reshape(-1)


class _RFOutcomeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        rf_ntrees: int = 800,
        rf_minleaf: int = 1,
        rf_maxfeat=1.0,
        rf_bootstrap: bool = True,
        rf_max_depth=None,
        rf_n_jobs: int = 1,
        seed: int = 0,
    ):
        self.rf_ntrees = int(rf_ntrees)
        self.rf_minleaf = int(rf_minleaf)
        self.rf_maxfeat = rf_maxfeat
        self.rf_bootstrap = bool(rf_bootstrap)
        self.rf_max_depth = rf_max_depth
        self.rf_n_jobs = int(rf_n_jobs)
        self.seed = int(seed)

    def fit(self, X, y):
        self.model_ = RandomForestRegressor(
            n_estimators=self.rf_ntrees,
            min_samples_leaf=self.rf_minleaf,
            max_features=self.rf_maxfeat,
            bootstrap=self.rf_bootstrap,
            max_depth=self.rf_max_depth,
            n_jobs=self.rf_n_jobs,
            random_state=self.seed,
        )
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1))
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)


class _PooledNNOutcomeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dims=(64, 32),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 400,
        patience: int = 25,
        min_delta: float = 1e-3,
        validation_fraction: float = 0.1,
        seed: int = 0,
    ):
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("NN outcome model expects design matrix with covariates plus treatment column.")

        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_dims,
            activation="relu",
            solver="adam",
            alpha=self.weight_decay,
            batch_size=int(max(16, self.batch_size)),
            learning_rate_init=self.lr,
            max_iter=self.max_epochs,
            shuffle=True,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.patience,
            tol=self.min_delta,
        )
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = self.scaler_.transform(X)
        return np.asarray(self.model_.predict(X_scaled), dtype=float).reshape(-1)


def _tmle_from_predictions(
    mu0,
    mu1,
    A,
    y,
    p1,
    *,
    true_ate=None,
    alpha=0.05,
    se_method: str = "if",
):
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=int).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    p1 = np.asarray(p1, dtype=float).reshape(-1)
    n = y.shape[0]
    if not (mu0.shape[0] == mu1.shape[0] == A.shape[0] == p1.shape[0] == n):
        raise ValueError("TMLE inputs must all have the same length.")

    p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
    p0 = 1.0 - p1
    q_aw = np.where(A == 1, mu1, mu0)
    H = (A / p1) - ((1 - A) / p0)
    denom = float(np.sum(H**2))
    epsilon = 0.0 if denom <= 0 else float(np.sum(H * (y - q_aw)) / denom)

    q1_star = mu1 + (epsilon / p1)
    q0_star = mu0 - (epsilon / p0)
    q_aw_star = np.where(A == 1, q1_star, q0_star)
    ate_hat = float(np.mean(q1_star - q0_star))

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan
    if se_method != "if":
        raise ValueError("se_method must be 'if' or 'none'")

    ic = H * (y - q_aw_star) + q1_star - q0_star - ate_hat
    se_hat = float(np.sqrt(np.sum(ic**2) / (n * (n - 1)))) if n > 1 else float("nan")

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat) if np.isfinite(se_hat) else float("nan")
    ci_hi = float(ate_hat + z * se_hat) if np.isfinite(se_hat) else float("nan")
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover


def _aipw_from_predictions(
    mu0,
    mu1,
    A,
    y,
    p1,
    *,
    true_ate=None,
    alpha=0.05,
    se_method: str = "if",
):
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    mu1 = np.asarray(mu1, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=int).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    p1 = np.asarray(p1, dtype=float).reshape(-1)
    n = y.shape[0]
    if not (mu0.shape[0] == mu1.shape[0] == A.shape[0] == p1.shape[0] == n):
        raise ValueError("AIPW inputs must all have the same length.")

    p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
    p0 = 1.0 - p1
    psi = mu1 - mu0 + (A / p1) * (y - mu1) - ((1 - A) / p0) * (y - mu0)
    ate_hat = float(np.mean(psi))

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan
    if se_method != "if":
        raise ValueError("se_method must be 'if' or 'none'")

    psi_c = psi - ate_hat
    se_hat = float(np.sqrt(np.sum(psi_c**2) / (n * (n - 1)))) if n > 1 else float("nan")

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat) if np.isfinite(se_hat) else float("nan")
    ci_hi = float(ate_hat + z * se_hat) if np.isfinite(se_hat) else float("nan")
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover


def _zepid_tmle(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    outcome_model,
    shared_outcome_fit=None,
    se_method: str = "if",  # "if" | "none"
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X = X_df.to_numpy(dtype=float)
    n = X.shape[0]
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    p1, exp_model = _resolve_propensity_inputs(
        X=X,
        n=n,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        seed=seed,
    )

    if shared_outcome_fit is not None:
        mu0 = np.asarray(shared_outcome_fit["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared_outcome_fit["mu1"], dtype=float).reshape(-1)
        if mu0.shape[0] != n or mu1.shape[0] != n:
            raise ValueError(
                f"Shared outcome fit length mismatch for TMLE: expected {n}, got "
                f"mu0={mu0.shape[0]} and mu1={mu1.shape[0]}."
            )
        return _tmle_from_predictions(
            mu0,
            mu1,
            A,
            y,
            p1,
            true_ate=true_ate,
            alpha=alpha,
            se_method=se_method,
        )

    from zepid.causal.doublyrobust import TMLE

    df_tmle = X_df.reset_index(drop=True).copy()
    exposure_name = "A"
    outcome_name = "__y__"
    df_tmle[exposure_name] = A
    df_tmle[outcome_name] = y

    cov_formula = _tmle_covariate_formula(X_df, exposure_name=exposure_name)
    tmle = TMLE(df_tmle, exposure=exposure_name, outcome=outcome_name, alpha=alpha)
    tmle.exposure_model(cov_formula, custom_model=exp_model, bound=trim, print_results=False)
    tmle.outcome_model(
        cov_formula,
        custom_model=outcome_model,
        bound=False,
        print_results=False,
        continuous_distribution="gaussian",
    )
    tmle.fit()
    ate_hat = float(tmle.average_treatment_effect)

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan
    if se_method != "if":
        raise ValueError("se_method must be 'if' or 'none'")

    se_hat = float(tmle.average_treatment_effect_se)
    ci_lo = float(tmle.average_treatment_effect_ci[0])
    ci_hi = float(tmle.average_treatment_effect_ci[1])
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover


def gb_gb_tmle(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    shared_outcome_fit=None,
    se_method: str = "if",
):
    outcome_model = _HistGBOutcomeRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
    )
    return _zepid_tmle(
        X_df,
        A_np,
        y,
        true_ate=true_ate,
        seed=seed,
        alpha=alpha,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        outcome_model=outcome_model,
        shared_outcome_fit=shared_outcome_fit,
        se_method=se_method,
    )


def gb_rf_tmle(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    rf_ntrees=800,
    rf_minleaf=1,
    rf_maxfeat=1.0,
    rf_bootstrap=True,
    rf_max_depth=None,
    rf_n_jobs=1,
    shared_outcome_fit=None,
    se_method: str = "if",
):
    outcome_model = _RFOutcomeRegressor(
        rf_ntrees=rf_ntrees,
        rf_minleaf=rf_minleaf,
        rf_maxfeat=rf_maxfeat,
        rf_bootstrap=rf_bootstrap,
        rf_max_depth=rf_max_depth,
        rf_n_jobs=rf_n_jobs,
        seed=seed,
    )
    return _zepid_tmle(
        X_df,
        A_np,
        y,
        true_ate=true_ate,
        seed=seed,
        alpha=alpha,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        outcome_model=outcome_model,
        shared_outcome_fit=shared_outcome_fit,
        se_method=se_method,
    )


def gb_nn_tmle(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    hidden_dims=(64, 32),
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=400,
    patience=25,
    min_delta=1e-3,
    shared_outcome_fit=None,
    se_method: str = "if",
):
    outcome_model = _PooledNNOutcomeRegressor(
        hidden_dims=hidden_dims,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
    )
    return _zepid_tmle(
        X_df,
        A_np,
        y,
        true_ate=true_ate,
        seed=seed,
        alpha=alpha,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        outcome_model=outcome_model,
        shared_outcome_fit=shared_outcome_fit,
        se_method=se_method,
    )


def gb_ols2_tmle(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    degree: int = 2,
    include_bias: bool = False,
    shared_outcome_fit=None,
    se_method: str = "if",
):
    """TMLE with the quadratic OLS outcome learner and GB propensity score."""
    outcome_model = _PolynomialOLSOutcomeRegressor(
        degree=int(degree),
        include_bias=bool(include_bias),
    )
    return _zepid_tmle(
        X_df,
        A_np,
        y,
        true_ate=true_ate,
        seed=seed,
        alpha=alpha,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        outcome_model=outcome_model,
        shared_outcome_fit=shared_outcome_fit,
        se_method=se_method,
    )


def gb_tmle(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_depth: int | None = 6,
    shared_outcome_fit=None,
    se_method: str = "if",
):
    return gb_gb_tmle(
        X_df,
        A_np,
        y,
        true_ate=true_ate,
        seed=seed,
        alpha=alpha,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        shared_outcome_fit=shared_outcome_fit,
        se_method=se_method,
    )


def gb_rf_aipw(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    # outcome (RF) hyperparams
    rf_ntrees=800,
    rf_minleaf=1,
    rf_maxfeat=1.0,
    rf_bootstrap=True,
    rf_max_depth=None,
    rf_n_jobs=1,
    shared_outcome_fit=None,
    # SE control
    se_method: str = "if",  # "if" | "none"
):
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X = X_df.to_numpy(dtype=float)
    n = X.shape[0]
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    p1, _ = _resolve_propensity_inputs(
        X=X,
        n=n,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        seed=seed,
    )
    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="rf", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        X_all = X_df.reset_index(drop=True).copy()
        X_all["A"] = A

        rf = RandomForestRegressor(
            n_estimators=int(rf_ntrees),
            min_samples_leaf=int(rf_minleaf),
            max_features=rf_maxfeat,
            bootstrap=bool(rf_bootstrap),
            max_depth=rf_max_depth,
            n_jobs=int(rf_n_jobs),
            random_state=int(seed),
        )
        rf.fit(X_all, y)

        X0 = X_df.reset_index(drop=True).copy()
        X0["A"] = 0
        X1 = X_df.reset_index(drop=True).copy()
        X1["A"] = 1
        mu0 = rf.predict(X0).astype(float).reshape(-1)
        mu1 = rf.predict(X1).astype(float).reshape(-1)

    return _aipw_from_predictions(
        mu0,
        mu1,
        A,
        y,
        p1,
        true_ate=true_ate,
        alpha=alpha,
        se_method=se_method,
    )


def gb_ols2_aipw(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    trim=0.01,
    p1_fixed=None,
    ps_model=None,
    shared_propensity_fit=None,
    degree: int = 2,
    include_bias: bool = False,
    shared_outcome_fit=None,
    se_method: str = "if",
):
    """AIPW with the quadratic OLS outcome learner and GB propensity score."""
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    X = X_df.to_numpy(dtype=float)
    n = X.shape[0]
    if y.shape[0] != n or A.shape[0] != n:
        raise ValueError(f"Length mismatch: n={n}, len(y)={len(y)}, len(A)={len(A)}")

    p1, _ = _resolve_propensity_inputs(
        X=X,
        n=n,
        trim=trim,
        p1_fixed=p1_fixed,
        ps_model=ps_model,
        shared_propensity_fit=shared_propensity_fit,
        seed=seed,
    )
    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="ols2", n=n)
    if shared is not None:
        mu0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        mu1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
    else:
        X_all, _, _ = _build_outcome_designs(X_df, A)
        model = _PolynomialOLSOutcomeRegressor(
            degree=int(degree),
            include_bias=bool(include_bias),
        )
        model.fit(np.asarray(X_all, dtype=float), y)
        mu0, mu1 = _predict_counterfactuals_from_shared_model(model, X_df)

    return _aipw_from_predictions(
        mu0,
        mu1,
        A,
        y,
        p1,
        true_ate=true_ate,
        alpha=alpha,
        se_method=se_method,
    )

# ==============================================================================
# Stochtree BART / XBART / BCF g-computation estimators
# ==============================================================================

def _prepare_stochtree_covariates(X_df: pd.DataFrame) -> pd.DataFrame:
    X_df_local = X_df.copy()
    binary_categories = [0, 1]

    for col in X_df_local.columns:
        vals = X_df_local[col].dropna().unique()
        if len(vals) == 2 and set(vals).issubset({0, 1}):
            # Keep binary levels stable across train / predict frames.
            X_df_local[col] = pd.Categorical(
                X_df_local[col],
                categories=binary_categories,
                ordered=True,
            )

    return X_df_local


def _fit_shared_bcf_outcome(
    X_df,
    A,
    y,
    *,
    seed: int,
    p1_fixed,
    num_gfr: int,
    num_burnin: int,
    num_mcmc: int,
    prognostic_trees: int,
    treatment_trees: int,
    prognostic_alpha: float,
    prognostic_beta: float,
    treatment_alpha: float,
    treatment_beta: float,
    standardize: bool,
    num_threads: int,
    num_chains: int,
    heteroskedastic: bool,
    var_trees: int,
    min_propensity: float,
):
    from stochtree import BCFModel

    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=int).reshape(-1)
    n = len(A)
    X = _prepare_stochtree_covariates(X_df)
    p_train = np.full(n, float(np.mean(A)), dtype=float) if p1_fixed is None else np.asarray(p1_fixed, dtype=float).reshape(-1)
    if len(p_train) != n:
        raise ValueError(f"Length mismatch: len(p_train)={len(p_train)}, expected {n}")
    p_train = np.clip(p_train, float(min_propensity), 1.0 - float(min_propensity))

    Z0 = np.zeros_like(A, dtype=int)
    Z1 = np.ones_like(A, dtype=int)
    model = BCFModel()
    general_params = {
        "num_threads": int(num_threads),
        "num_chains": int(num_chains),
        "random_seed": int(seed),
        "standardize": bool(standardize),
    }
    prognostic_forest_params = {
        "num_trees": int(prognostic_trees),
        "alpha": float(prognostic_alpha),
        "beta": float(prognostic_beta),
    }
    treatment_effect_forest_params = {
        "num_trees": int(treatment_trees),
        "alpha": float(treatment_alpha),
        "beta": float(treatment_beta),
    }
    variance_forest_params = None
    if heteroskedastic:
        variance_forest_params = {
            "num_trees": int(var_trees),
            "alpha": float(prognostic_alpha),
            "beta": float(prognostic_beta),
        }

    model.sample(
        X_train=X,
        Z_train=A,
        y_train=y,
        propensity_train=p_train,
        num_gfr=int(num_gfr),
        num_burnin=int(num_burnin),
        num_mcmc=int(num_mcmc),
        general_params=general_params,
        prognostic_forest_params=prognostic_forest_params,
        treatment_effect_forest_params=treatment_effect_forest_params,
        variance_forest_params=variance_forest_params,
    )

    mu1 = np.asarray(model.predict(X=X, Z=Z1, propensity=p_train, terms="y_hat", type="mean"), dtype=float).reshape(-1)
    mu0 = np.asarray(model.predict(X=X, Z=Z0, propensity=p_train, terms="y_hat", type="mean"), dtype=float).reshape(-1)
    return {
        "family": "bcf",
        "model": model,
        "mu0": mu0,
        "mu1": mu1,
        "p_train": p_train,
    }


def bcf_gcomp(
    X_df,
    A_np,
    y,
    *,
    true_ate=None,
    seed=0,
    alpha=0.05,
    p1_fixed=None,
    num_gfr=10,
    num_burnin=20,
    num_mcmc=30,
    prognostic_trees=250,
    treatment_trees=50,
    prognostic_alpha=0.95,
    prognostic_beta=2.0,
    treatment_alpha=0.25,
    treatment_beta=3.0,
    standardize=True,
    num_threads=0,
    num_chains=1,
    heteroskedastic=False,
    var_trees=20,
    shared_outcome_fit=None,
    se_method="if",
    min_propensity=1e-3,
):
    """
    Stochtree BCF g-computation:
      1) Fit a causal BCFModel to (X, A, y), optionally using fixed propensities.
      2) Predict yhat(X,1) and yhat(X,0) from the fitted causal model.
      3) ATE_hat = mean_i [ yhat(X_i,1) - yhat(X_i,0) ].

    When `se_method="if"`, we use an AIPW-style influence-function plug-in with
    the BCF mean predictions and supplied / empirical propensities.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    A = np.asarray(A_np, dtype=int).reshape(-1)
    n = len(A)
    if len(X_df) != n or len(y) != n:
        raise ValueError(f"Length mismatch: n={n}, len(X_df)={len(X_df)}, len(y)={len(y)}")

    shared = _validate_shared_outcome_fit(shared_outcome_fit, family="bcf", n=n)
    if shared is not None:
        y0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        y1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
        p_train = np.asarray(shared["p_train"], dtype=float).reshape(-1)
    else:
        shared = _fit_shared_bcf_outcome(
            X_df,
            A,
            y,
            seed=int(seed),
            p1_fixed=p1_fixed,
            num_gfr=int(num_gfr),
            num_burnin=int(num_burnin),
            num_mcmc=int(num_mcmc),
            prognostic_trees=int(prognostic_trees),
            treatment_trees=int(treatment_trees),
            prognostic_alpha=float(prognostic_alpha),
            prognostic_beta=float(prognostic_beta),
            treatment_alpha=float(treatment_alpha),
            treatment_beta=float(treatment_beta),
            standardize=bool(standardize),
            num_threads=int(num_threads),
            num_chains=int(num_chains),
            heteroskedastic=bool(heteroskedastic),
            var_trees=int(var_trees),
            min_propensity=float(min_propensity),
        )
        y0 = np.asarray(shared["mu0"], dtype=float).reshape(-1)
        y1 = np.asarray(shared["mu1"], dtype=float).reshape(-1)
        p_train = np.asarray(shared["p_train"], dtype=float).reshape(-1)
    tau = y1 - y0
    ate_hat = float(np.mean(tau))

    if se_method == "none":
        return ate_hat, np.nan, np.nan, np.nan, np.nan
    if se_method != "if":
        raise ValueError("se_method must be 'if' or 'none'")

    psi = tau + (A / p_train) * (y - y1) - ((1 - A) / (1.0 - p_train)) * (y - y0)
    psi_c = psi - float(np.mean(psi))
    se_hat = float(np.sqrt(np.sum(psi_c**2) / (n * (n - 1)))) if n > 1 else float("nan")

    z = float(norm.ppf(1 - alpha / 2))
    ci_lo = float(ate_hat - z * se_hat) if np.isfinite(se_hat) else float("nan")
    ci_hi = float(ate_hat + z * se_hat) if np.isfinite(se_hat) else float("nan")
    cover = float(
        (true_ate is not None)
        and np.isfinite(ci_lo) and np.isfinite(ci_hi)
        and (true_ate >= ci_lo) and (true_ate <= ci_hi)
    )
    return ate_hat, se_hat, ci_lo, ci_hi, cover
