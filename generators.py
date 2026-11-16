
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

def _as_1d_float(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape; got shape {a.shape}")
    return a


def _as_1d_int01(x, name: str) -> np.ndarray:
    a = np.asarray(x).reshape(-1)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    a = np.rint(a).astype(int)
    u = np.unique(a)
    if not set(u.tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must be binary {{0,1}}; got unique values {u}")
    return a


@dataclass
class NoiseSpec:
    noise_type: str = "normal"  # "normal" or "t"
    t_df: int = 7               # used only if noise_type == "t"

    def __post_init__(self):
        self.noise_type = str(self.noise_type)
        self.t_df = int(self.t_df)
        if self.noise_type not in ("normal", "t"):
            raise ValueError("noise_type must be 'normal' or 't'")
        if self.noise_type == "t" and self.t_df <= 2:
            raise ValueError("t_df must be > 2 for finite variance (t noise).")

    def eps(self, rng: np.random.Generator, shape):
        if self.noise_type == "normal":
            return rng.standard_normal(shape)
        raw = rng.standard_t(self.t_df, size=shape)
        return raw / np.sqrt(self.t_df / (self.t_df - 2))


class RealScoreNoiseDGP:
    """
    REAL DGP:
        y = score + eps,  eps ~ Normal(0, noise_sd^2)
    score is fixed given X; randomness only from eps.
    """

    def __init__(self, score, noise_sd: float):
        self.score = _as_1d_float(score, "score")
        self.n = int(self.score.size)
        self.noise_sd = float(noise_sd)
        if not np.isfinite(self.noise_sd) or self.noise_sd < 0:
            raise ValueError(f"noise_sd must be finite and >=0; got {self.noise_sd}")

    def __call__(self, seed: int):
        rng = np.random.default_rng(int(seed))
        eps = rng.normal(0.0, self.noise_sd, size=self.n)
        return (self.score + eps).reshape(-1)


class CachedOutcomeGenerator:
    """
    Generator for observed outcomes given fixed observed treatment A_vec.
    Requires cached mu0, mu1, sigma0, sigma1 arrays (length n).

    Observed outcome draw:
      y0 = mu0 + sigma0 * eps0
      y1 = mu1 + sigma1 * eps1
      y  = y1 if A=1 else y0
    """

    def __init__(
        self,
        mu0,
        mu1,
        sigma0,
        sigma1,
        A_vec,
        *,
        noise_type: str = "normal",
        t_df: int = 7,
        clip_sigma_min: float | None = 0.0,
    ):
        self.mu0 = _as_1d_float(mu0, "mu0")
        self.mu1 = _as_1d_float(mu1, "mu1")
        self.sigma0 = _as_1d_float(sigma0, "sigma0")
        self.sigma1 = _as_1d_float(sigma1, "sigma1")
        self.A_vec = _as_1d_int01(A_vec, "A_vec")
        self.n = int(self.mu0.shape[0])

        if any(arr.shape[0] != self.n for arr in (self.mu1, self.sigma0, self.sigma1, self.A_vec)):
            raise ValueError(
                "All cached arrays must have the same length n. "
                f"Got: mu0 {self.mu0.shape}, mu1 {self.mu1.shape}, "
                f"s0 {self.sigma0.shape}, s1 {self.sigma1.shape}, A {self.A_vec.shape}"
            )

        self.noise = NoiseSpec(noise_type=noise_type, t_df=t_df)

        if clip_sigma_min is not None:
            m = float(clip_sigma_min)
            if m < 0:
                raise ValueError("clip_sigma_min must be >= 0 or None.")
            self.sigma0 = np.maximum(self.sigma0, m)
            self.sigma1 = np.maximum(self.sigma1, m)

    def draw_y0(self, seed: int, n_samples: int = 1):
        rng = np.random.default_rng(int(seed))
        S = int(n_samples)
        eps = self.noise.eps(rng, (self.n, S))
        out = self.mu0[:, None] + self.sigma0[:, None] * eps
        return out[:, 0] if S == 1 else out

    def draw_y1(self, seed: int, n_samples: int = 1):
        rng = np.random.default_rng(int(seed))
        S = int(n_samples)
        eps = self.noise.eps(rng, (self.n, S))
        out = self.mu1[:, None] + self.sigma1[:, None] * eps
        return out[:, 0] if S == 1 else out

    def __call__(self, seed: int):
        rng = np.random.default_rng(int(seed))
        eps0 = self.noise.eps(rng, (self.n,))
        eps1 = self.noise.eps(rng, (self.n,))
        y0 = self.mu0 + self.sigma0 * eps0
        y1 = self.mu1 + self.sigma1 * eps1
        return np.where(self.A_vec == 1, y1, y0).reshape(-1)


class RFGenerator(CachedOutcomeGenerator):
    pass


class GBGenerator(CachedOutcomeGenerator):
    pass


class NNGenerator(CachedOutcomeGenerator):
    pass


class OLSGenerator(CachedOutcomeGenerator):
    pass




class WGANGenerator:
    """
    Wrap a fitted WGAN object so it behaves like the other generators.

    Expected fitted_wgan interface:
      - sample_y0(X, n_samples=1, seed=None) -> (n,) or (n,S)
      - sample_y1(X, n_samples=1, seed=None) -> (n,) or (n,S)
    """

    def __init__(self, fitted_wgan=None, X=None, A_vec=None, estimator_payload=None):
        if X is None or A_vec is None:
            raise ValueError("X and A_vec are required for WGANGenerator")
        if fitted_wgan is None and estimator_payload is None:
            raise ValueError("Provide either fitted_wgan or estimator_payload")

        self.fitted_wgan = fitted_wgan
        self.estimator_payload = estimator_payload
        self.X = np.asarray(X, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {self.X.shape}")

        self.A_vec = _as_1d_int01(A_vec, "A_vec")
        self.n = int(self.X.shape[0])
        self._X_tensor = None

        if self.A_vec.shape[0] != self.n:
            raise ValueError(
                f"Length mismatch: X has {self.n} rows but A_vec has {self.A_vec.shape[0]}"
            )

        if self.fitted_wgan is not None:
            covariate_scaler = self.fitted_wgan.covariate_scaler
            self.noise_dim = int(self.fitted_wgan.noise_dim)
            self.outcome_scaler_0 = self.fitted_wgan.outcome_scaler_0
            self.outcome_scaler_1 = self.fitted_wgan.outcome_scaler_1
        else:
            scalers = self.estimator_payload["scalers"]
            covariate_scaler = scalers["covariate_scaler"]
            self.noise_dim = int(self.estimator_payload["config"]["noise_dim"])
            self.outcome_scaler_0 = scalers["outcome_scaler_0"]
            self.outcome_scaler_1 = scalers["outcome_scaler_1"]

        self.X_scaled = np.asarray(
            covariate_scaler.transform(self.X),
            dtype=np.float32,
        )

    def _get_wgan(self):
        if self.fitted_wgan is None:
            if self.estimator_payload is None:
                raise RuntimeError("WGANGenerator has neither an in-memory model nor an estimator payload")
            from wgan_module import load_estimator_payload
            self.fitted_wgan = load_estimator_payload(self.estimator_payload, map_location="cpu")
        return self.fitted_wgan

    def _get_X_tensor(self):
        if self._X_tensor is None:
            import torch
            self._X_tensor = torch.tensor(self.X_scaled, dtype=torch.float32)
        return self._X_tensor

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("estimator_payload") is not None:
            state["fitted_wgan"] = None
        state["_X_tensor"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def draw_y0(self, seed: int, n_samples: int = 1):
        wgan = self._get_wgan()
        import torch

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        X_t = self._get_X_tensor()
        out_scaled = wgan.wgan_control.generate(
            X_t,
            noise_dim=self.noise_dim,
            n_samples=int(n_samples),
        ).detach().cpu().numpy()
        out = self.outcome_scaler_0.inverse_transform(
            out_scaled.reshape(-1, 1)
        ).reshape(self.n, int(n_samples))

        if int(n_samples) == 1:
            return out[:, 0]

        if out.shape != (self.n, int(n_samples)):
            raise ValueError(
                f"WGAN sample_y0 returned shape {out.shape}, expected {(self.n, int(n_samples))}"
            )
        return out

    def draw_y1(self, seed: int, n_samples: int = 1):
        wgan = self._get_wgan()
        import torch

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        X_t = self._get_X_tensor()
        out_scaled = wgan.wgan_treated.generate(
            X_t,
            noise_dim=self.noise_dim,
            n_samples=int(n_samples),
        ).detach().cpu().numpy()
        out = self.outcome_scaler_1.inverse_transform(
            out_scaled.reshape(-1, 1)
        ).reshape(self.n, int(n_samples))

        if int(n_samples) == 1:
            return out[:, 0]

        if out.shape != (self.n, int(n_samples)):
            raise ValueError(
                f"WGAN sample_y1 returned shape {out.shape}, expected {(self.n, int(n_samples))}"
            )
        return out

    def __call__(self, seed: int):
        y0 = np.asarray(self.draw_y0(seed=int(seed), n_samples=1), dtype=float).reshape(-1)
        y1 = np.asarray(self.draw_y1(seed=int(seed) + 1, n_samples=1), dtype=float).reshape(-1)

        if y0.shape[0] != self.n or y1.shape[0] != self.n:
            raise ValueError(
                f"WGAN returned wrong length: y0 {y0.shape}, y1 {y1.shape}, expected ({self.n},)"
            )

        return np.where(self.A_vec == 1, y1, y0).reshape(-1)


class PreGeneratedObservedGenerator:
    """
    Lightweight spawn-safe generator that serves pre-generated observed outcomes.

    Column j of y_draws corresponds to Monte Carlo replicate seed_base + j.
    """

    def __init__(self, y_draws, seed_base: int):
        arr = np.asarray(y_draws, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"y_draws must be 2D; got shape {arr.shape}")
        self.y_draws = arr
        self.seed_base = int(seed_base)
        self.n = int(arr.shape[0])
        self.n_mc = int(arr.shape[1])

    def __call__(self, seed: int):
        idx = int(seed) - self.seed_base
        if idx < 0 or idx >= self.n_mc:
            raise IndexError(
                f"Requested seed {seed} maps to column {idx}, but only {self.n_mc} draws are stored"
            )
        return self.y_draws[:, idx].reshape(-1).copy()
