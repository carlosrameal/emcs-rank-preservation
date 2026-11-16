from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ACIC2017_DATA_PATH = Path("datasets") / "IHDP_ACIC2017" / "acic2017_design.parquet"
NIE_DGP_CODES = ("NieA", "NieB", "NieC", "NieD")


def selected_dgp_code(args) -> str:
    """Return the explicit DGP selector, preserving the legacy ACIC arguments."""
    explicit = getattr(args, "dgp", None)
    if explicit:
        return str(explicit)
    return f"ACIC{str(getattr(args, 'acic_version', '2017'))}"


def is_nie_dgp(args) -> bool:
    return selected_dgp_code(args) in NIE_DGP_CODES


def resolve_default_data_path(*, project_root: Path, acic_version: str = "2017") -> Path:
    return project_root / DEFAULT_ACIC2017_DATA_PATH


def build_structured_dgp_label(args, *, data_path: str | Path) -> str:
    if args.dgp_label:
        return str(args.dgp_label)

    dgp_code = selected_dgp_code(args)
    if dgp_code in NIE_DGP_CODES:
        return (
            f"{dgp_code.lower()}"
            f"__n{int(args.synthetic_n)}"
            f"__d{int(args.synthetic_d)}"
            f"__sigma{float(args.synthetic_sigma):g}"
        )

    stem = Path(data_path).stem or "dataset"
    label = (
        f"{stem}__acic2017__{args.acic_error}"
        f"__mag{int(args.acic_magnitude)}"
        f"__noise{int(args.acic_noise)}"
        f"__conf{int(args.acic_confounding)}"
    )
    if str(args.acic_treatment_mode) == "redraw":
        label += "__tredraw"
    return label


def structured_mode_uses_shared_cache(args) -> bool:
    if is_nie_dgp(args):
        # Each observed dataset has newly generated covariates and treatment.
        return False
    return str(args.acic_treatment_mode) != "redraw"


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    positive = x >= 0.0
    out = np.empty_like(x, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _require_nie_dimension(dimension: int) -> int:
    dimension = int(dimension)
    if dimension < 5:
        raise ValueError("NieA--NieD require --synthetic-d to be at least 5.")
    return dimension


def build_nie_surfaces(X_df: pd.DataFrame, *, dgp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return propensity, baseline effect, and CATE for the four Nie designs."""
    dgp = str(dgp)
    if dgp not in NIE_DGP_CODES:
        raise ValueError(f"Unknown Nie DGP '{dgp}'. Expected one of {NIE_DGP_CODES}.")
    if X_df.shape[1] < 5:
        raise ValueError("NieA--NieD require at least five covariates.")

    x = X_df.to_numpy(dtype=float)
    x1, x2, x3, x4, x5 = (x[:, j] for j in range(5))
    relu = lambda z: np.maximum(z, 0.0)

    if dgp == "NieA":
        propensity = np.clip(np.sin(np.pi * x1 * x2), 0.1, 0.9)
        baseline = np.sin(np.pi * x1 * x2) + 2.0 * (x3 - 0.5) ** 2 + x4 + 0.5 * x5
        cate = (x1 + x2) / 2.0
    elif dgp == "NieB":
        propensity = np.full(x.shape[0], 0.5, dtype=float)
        baseline = relu(np.maximum(x1 + x2, x3)) + relu(x4 + x5)
        cate = x1 + np.logaddexp(0.0, x2)
    elif dgp == "NieC":
        propensity = _expit(-(x2 + x3))
        baseline = 2.0 * np.logaddexp(0.0, x1 + x2 + x3)
        cate = np.ones(x.shape[0], dtype=float)
    else:  # NieD
        propensity = 1.0 / (1.0 + np.exp(-x1) + np.exp(-x2))
        arm0 = relu(x1 + x2 + x3)
        arm1 = relu(x4 + x5)
        baseline = (arm0 + arm1) / 2.0
        cate = arm0 - arm1

    return (
        np.asarray(propensity, dtype=float).reshape(-1),
        np.asarray(baseline, dtype=float).reshape(-1),
        np.asarray(cate, dtype=float).reshape(-1),
    )


def generate_nie_observed_data(
    *,
    dgp: str,
    n: int,
    dimension: int,
    sigma: float,
    seed: int,
) -> dict:
    """Generate one observed dataset from NieA--NieD.

    Outcomes follow Y=b(X)+(A-0.5)tau(X)+sigma*epsilon, epsilon~N(0,1).
    """
    n = int(n)
    dimension = _require_nie_dimension(dimension)
    sigma = float(sigma)
    if n <= 0:
        raise ValueError("--synthetic-n must be positive.")
    if sigma < 0.0:
        raise ValueError("--synthetic-sigma must be non-negative.")

    rng = np.random.default_rng(int(seed))
    if dgp == "NieA":
        x = rng.uniform(0.0, 1.0, size=(n, dimension))
    else:
        x = rng.normal(0.0, 1.0, size=(n, dimension))
    X_df = pd.DataFrame(x, columns=[f"x_{j + 1}" for j in range(dimension)])
    propensity, baseline, cate = build_nie_surfaces(X_df, dgp=dgp)
    A_binary = rng.binomial(1, propensity, size=n).astype(int)
    epsilon = rng.normal(0.0, 1.0, size=n)
    y = baseline + (A_binary - 0.5) * cate + sigma * epsilon
    return {
        "X_df": X_df,
        "A_binary": A_binary,
        "Y_parts": {
            "y": y,
            "alpha": cate,
            "baseline": baseline,
            "p_true": propensity,
            "sigma": np.full(n, sigma, dtype=float),
        },
    }


class NieObservedDGPSampler:
    """Fixed-design redrawing sampler for a realized Nie observed dataset."""

    def __init__(self, *, baseline: np.ndarray, cate: np.ndarray, sigma: float, fixed_a: np.ndarray):
        self.baseline = np.asarray(baseline, dtype=float).reshape(-1)
        self.cate = np.asarray(cate, dtype=float).reshape(-1)
        self.sigma = float(sigma)
        self.fixed_a = np.asarray(fixed_a, dtype=int).reshape(-1)

    def draw_observed(self, *, seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        eps = rng.normal(0.0, 1.0, size=(self.fixed_a.size, int(n_samples)))
        y = (
            self.baseline[:, None]
            + (self.fixed_a[:, None] - 0.5) * self.cate[:, None]
            + self.sigma * eps
        )
        a = np.repeat(self.fixed_a[:, None], int(n_samples), axis=1)
        return y, a


def _acic_treatment_coefficients(confounding: int) -> tuple[float, float]:
    if int(confounding) <= 0:
        return 0.0, 0.5
    return -1.0, 3.0


def _acic_treatment_signal(X_df: pd.DataFrame) -> np.ndarray:
    required = ["x_1", "x_43", "x_10_term"]
    missing = [c for c in required if c not in X_df.columns]
    if missing:
        raise ValueError(f"Missing required ACIC treatment columns: {missing}")

    return (
        X_df["x_1"].to_numpy(dtype=float)
        + X_df["x_43"].to_numpy(dtype=float)
        + 0.3 * X_df["x_10_term"].to_numpy(dtype=float)
    )


def compute_acic2017_propensity(X_df: pd.DataFrame, *, confounding: int) -> np.ndarray:
    beta0, beta1 = _acic_treatment_coefficients(int(confounding))
    f_x = _acic_treatment_signal(X_df)
    p_true = 1.0 / (1.0 + np.exp(beta0 + beta1 * f_x))
    return np.asarray(p_true, dtype=float).reshape(-1)


def generate_acic2017_outcomes(
    X_df: pd.DataFrame,
    A,
    *,
    error="iid",
    magnitude=1,
    noise=1,
    confounding=1,
    seed=42,
    return_components=False,
):
    from scipy.stats import norm

    required = [
        "x_1", "x_43", "x_10_term",
        "x_3_leq0", "x_24_B",
        "x_14_term", "x_15_term",
        "x_21_code",
    ]
    missing = [c for c in required if c not in X_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in X_df: {missing}")

    rng = np.random.default_rng(seed)

    A = np.asarray(A).reshape(-1).astype(int)
    n = len(X_df)
    if len(A) != n:
        raise ValueError("Length of A must match X_df")

    effect_size = (1.0 / 3.0) if magnitude <= 0 else 2.0
    snr = 0.25 if noise <= 0 else 1.25
    reffectv = 0.1 if error == "group_corr" else 0.0

    p = compute_acic2017_propensity(X_df, confounding=int(confounding))
    mu = -np.sin(norm.ppf(p)) + X_df["x_43"].to_numpy(dtype=float)
    alpha = effect_size * (
        X_df["x_3_leq0"].to_numpy(dtype=float) * X_df["x_24_B"].to_numpy(dtype=float)
        + X_df["x_14_term"].to_numpy(dtype=float)
        - X_df["x_15_term"].to_numpy(dtype=float)
    )

    linpart = mu + p * alpha
    sigma_y = snr * np.std(linpart, ddof=1)

    if error == "heteroskedastic":
        het_grid = np.linspace(0.4, 1.4, 16)
        x21_idx = X_df["x_21_code"].to_numpy(dtype=int) - 1
        het = het_grid[x21_idx]
    else:
        het = np.ones(n)

    eps = rng.normal(0.0, 1.0, size=n)
    y_temp = mu + het * (1.0 - reffectv) * sigma_y * eps
    y_obs = y_temp + A * alpha

    if error == "group_corr":
        x21_codes = X_df["x_21_code"].to_numpy(dtype=int)
        n_groups = x21_codes.max()
        ref = reffectv * sigma_y * rng.normal(size=n_groups)
        y_obs = y_obs + ref[x21_codes - 1]

    if error == "nonadditive":
        b = np.sqrt(sigma_y**2 + np.var(linpart, ddof=1))
        a = np.mean(linpart)
        eps_small = rng.normal(0.0, 0.01, size=n)
        y_obs = 13.0 * norm.cdf(y_obs, loc=a, scale=1.25 * b) - 6.0 + eps_small

        muvec1 = (mu + alpha - a) / (1.25 * b)
        muvec0 = (mu - a) / (1.25 * b)
        alpha_true = (
            13.0 * norm.cdf(muvec1 / np.sqrt(sigma_y**2 / (1.25 * b)**2 + 1.0))
            - 13.0 * norm.cdf(muvec0 / np.sqrt(sigma_y**2 / (1.25 * b)**2 + 1.0))
        )
    else:
        alpha_true = alpha.copy()

    if return_components:
        return {
            "y": y_obs,
            "mu": mu,
            "alpha": alpha_true,
            "p": p,
            "sigma_y": sigma_y,
        }
    return y_obs


def generate_acic2017_treatment(
    X_df: pd.DataFrame,
    *,
    confounding: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    p_true = compute_acic2017_propensity(X_df, confounding=int(confounding))
    rng = np.random.default_rng(int(seed))
    A = rng.binomial(1, p_true, size=len(X_df)).astype(int).reshape(-1)
    return A, p_true


class ACIC2017ObservedDGPSampler:
    def __init__(
        self,
        *,
        X_df: pd.DataFrame,
        error: str,
        magnitude: int,
        noise: int,
        confounding: int,
        treatment_mode: str,
        fixed_a: np.ndarray | None,
    ):
        self.X_df = X_df
        self.error = str(error)
        self.magnitude = int(magnitude)
        self.noise = int(noise)
        self.confounding = int(confounding)
        self.treatment_mode = str(treatment_mode)
        self.fixed_a = None if fixed_a is None else np.asarray(fixed_a, dtype=int).reshape(-1)

    def _draw_one(self, *, treatment_seed: int, outcome_seed: int) -> tuple[np.ndarray, np.ndarray]:
        if self.treatment_mode == "redraw":
            a_rep, _ = generate_acic2017_treatment(
                self.X_df,
                confounding=self.confounding,
                seed=int(treatment_seed),
            )
        else:
            if self.fixed_a is None:
                raise ValueError("fixed_a must be provided when ACIC treatment mode is 'fixed'.")
            a_rep = np.asarray(self.fixed_a, dtype=int).reshape(-1).copy()

        y_rep = np.asarray(
            generate_acic2017_outcomes(
                self.X_df,
                a_rep,
                error=self.error,
                magnitude=self.magnitude,
                noise=self.noise,
                confounding=self.confounding,
                seed=int(outcome_seed),
                return_components=False,
            ),
            dtype=float,
        ).reshape(-1)
        return y_rep, a_rep

    def draw_observed(self, *, seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        y_cols = []
        a_cols = []
        seed = int(seed)
        for r in range(int(n_samples)):
            rep_seed = seed + 2 * int(r)
            y_rep, a_rep = self._draw_one(
                treatment_seed=rep_seed,
                outcome_seed=rep_seed + 1,
            )
            y_cols.append(y_rep)
            a_cols.append(a_rep)
        return np.column_stack(y_cols), np.column_stack(a_cols)


def generate_structured_observed_data(
    df: pd.DataFrame | None,
    args,
    *,
    seed_plan: dict[str, int],
) -> dict:
    dgp_code = selected_dgp_code(args)
    if dgp_code in NIE_DGP_CODES:
        generated = generate_nie_observed_data(
            dgp=dgp_code,
            n=int(args.synthetic_n),
            dimension=int(args.synthetic_d),
            sigma=float(args.synthetic_sigma),
            seed=int(seed_plan["job_seed"]),
        )
        X_df = generated["X_df"]
        A_binary = generated["A_binary"]
        Y_parts = generated["Y_parts"]
        sampler = NieObservedDGPSampler(
            baseline=Y_parts["baseline"],
            cate=Y_parts["alpha"],
            sigma=float(args.synthetic_sigma),
            fixed_a=A_binary,
        )
        return {
            "version": dgp_code,
            "X_df": X_df,
            "A_binary": A_binary,
            "Y_parts": Y_parts,
            "outcome_label": f"generated_{dgp_code.lower()}_outcome",
            "sampler": sampler,
            "uses_shared_cache": False,
            "metadata": {
                "outcome_mode": f"generated_{dgp_code.lower()}_outcome",
                "dgp": dgp_code,
                "synthetic_n": int(args.synthetic_n),
                "synthetic_d": int(args.synthetic_d),
                "synthetic_sigma": float(args.synthetic_sigma),
                "seed": int(seed_plan["job_seed"]),
                "benchmark_treatment_mode": "fixed",
            },
            "log_lines": [
                f"    dgp:          {dgp_code}",
                f"    n:            {int(args.synthetic_n)}",
                f"    dimension:    {int(args.synthetic_d)}",
                f"    sigma:        {float(args.synthetic_sigma):g}",
                "    treatment:    Bernoulli draw conditional on X (fixed for benchmarking)",
                f"    seed:         {seed_plan['job_seed']}",
            ],
        }

    if df is None:
        raise ValueError("An input dataframe is required for ACIC structured DGPs.")
    protected_cols = {c for c in [args.treatment_col, "p_true"] if c in df.columns}
    drop_cols = [c for c in args.drop_cols if c in df.columns and c not in protected_cols]
    base_drop_cols = [c for c in [args.treatment_col, "p_true"] if c in df.columns]
    version = "2017"

    if version == "2017":
        X_df = df.drop(columns=[*base_drop_cols, *drop_cols]).copy()
        if str(args.acic_treatment_mode) == "fixed":
            if args.treatment_col not in df.columns:
                raise ValueError(f"Missing treatment column '{args.treatment_col}' in ACIC 2017 design mode.")
            A_binary = np.rint(df[args.treatment_col].to_numpy(dtype=float).reshape(-1)).astype(int)
            if not set(np.unique(A_binary)).issubset({0, 1}):
                raise ValueError("Treatment column must be binary and coded as {0,1}.")
        else:
            A_binary, _ = generate_acic2017_treatment(
                X_df,
                confounding=int(args.acic_confounding),
                seed=int(seed_plan["acic_treatment"]),
            )
        Y_parts = generate_acic2017_outcomes(
            X_df,
            A_binary,
            error=args.acic_error,
            magnitude=int(args.acic_magnitude),
            noise=int(args.acic_noise),
            confounding=int(args.acic_confounding),
            seed=int(seed_plan["acic_outcome"]),
            return_components=True,
        )
        sampler = ACIC2017ObservedDGPSampler(
            X_df=X_df,
            error=args.acic_error,
            magnitude=int(args.acic_magnitude),
            noise=int(args.acic_noise),
            confounding=int(args.acic_confounding),
            treatment_mode="fixed",
            fixed_a=A_binary,
        )
        return {
            "version": version,
            "X_df": X_df,
            "A_binary": A_binary,
            "Y_parts": Y_parts,
            "outcome_label": "generated_acic2017_outcome",
            "sampler": sampler,
            "uses_shared_cache": structured_mode_uses_shared_cache(args),
            "metadata": {
                "data_path": str(Path(args.data_path).resolve()),
                "treatment_col": str(args.treatment_col),
                "outcome_mode": "generated_acic2017_outcome",
                "acic_version": version,
                "acic_treatment_mode": str(args.acic_treatment_mode),
                "acic_error": str(args.acic_error),
                "acic_magnitude": int(args.acic_magnitude),
                "acic_noise": int(args.acic_noise),
                "acic_confounding": int(args.acic_confounding),
            },
            "log_lines": [
                f"    version:      {version}",
                f"    treatment_mode:{str(args.acic_treatment_mode):>12}",
                f"    error:        {args.acic_error}",
                f"    magnitude:    {int(args.acic_magnitude)}",
                f"    noise:        {int(args.acic_noise)}",
                f"    confounding:  {int(args.acic_confounding)}",
                (
                    f"    treatment_seed:{seed_plan['acic_treatment']:>8}"
                    if str(args.acic_treatment_mode) == "redraw"
                    else "    treatment_seed:   observed A from design"
                ),
                f"    seed:         {seed_plan['acic_outcome']}",
            ],
        }
