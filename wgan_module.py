from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import multiprocessing as mp
import shutil
import tempfile
import time


def _safe_cpu_count_preimport() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(key)
        if value is not None:
            try:
                parsed = int(value)
                if parsed > 0:
                    return parsed
            except Exception:
                pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass

    return os.cpu_count() or 1


_PREIMPORT_NCPUS = _safe_cpu_count_preimport()
os.environ["OMP_NUM_THREADS"] = str(_PREIMPORT_NCPUS)
os.environ["MKL_NUM_THREADS"] = str(_PREIMPORT_NCPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_PREIMPORT_NCPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(_PREIMPORT_NCPUS)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from scipy.spatial.distance import jensenshannon
from scipy.stats import energy_distance, ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset


def _windows_available_cpus() -> int | None:
    if os.name != "nt":
        return None

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        current_process = kernel32.GetCurrentProcess()

        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        ok = kernel32.GetProcessAffinityMask(
            current_process,
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        )
        if ok:
            mask_value = int(process_mask.value)
            if mask_value > 0:
                return int(mask_value.bit_count())
    except Exception:
        pass

    return None


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

    windows_cpus = _windows_available_cpus()
    if windows_cpus is not None:
        return int(max(1, windows_cpus))

    if hasattr(os, "process_cpu_count"):
        try:
            process_cpus = os.process_cpu_count()
            if process_cpus is not None and int(process_cpus) > 0:
                return int(process_cpus)
        except Exception:
            pass

    return int(max(1, os.cpu_count() or 1))


def set_worker_threads(n_threads: int):
    n_threads = max(1, int(n_threads))
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)


def runtime_diagnostics() -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except Exception:
            affinity = None

    return {
        "available_cpus": available_cpus(),
        "cpu_affinity": affinity,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _clone_state_dict_cpu(module: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _arm_checkpoint_w1_score_worker(wgan, X_t, Y_t_scaled, *, noise_dim: int, n_draws: int, seed: int | None = None):
    y_real = Y_t_scaled.detach().cpu().numpy().reshape(-1)
    scores = []

    for draw_idx in range(int(n_draws)):
        if seed is not None:
            local_seed = int(seed) + int(draw_idx)
            torch.manual_seed(local_seed)
            np.random.seed(local_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(local_seed)

        y_fake = wgan.generate(
            X_t,
            noise_dim=noise_dim,
            n_samples=1,
        ).detach().cpu().numpy().reshape(-1)
        scores.append(float(wasserstein_distance(y_real, y_fake)))

    return float(np.mean(scores))


def _train_single_arm_worker(
    wgan,
    X_t,
    Y_t,
    *,
    noise_dim: int,
    n_epochs: int,
    print_every: int,
    batch_size: int,
    label: str,
    checkpoint_every: int,
    checkpoint_n_draws: int,
    worker_threads: int,
    device: str,
    base_seed: int,
    seed_offset: int,
    queue,
    save_dir,
):
    start_time = time.time()
    torch.manual_seed(int(base_seed))
    np.random.seed(int(base_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(base_seed))
    set_worker_threads(worker_threads)
    wgan.set_device(device)
    print(f"[{label}] worker started | device={device} | threads={worker_threads} | n={X_t.size(0)}", flush=True)

    dataset = TensorDataset(X_t, Y_t)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        pin_memory=True,
    )
    best_w1 = float("inf")
    best_epoch = None
    best_generator_state = None
    best_critic_state = None
    checkpoint_log = []
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    label_slug = str(label).strip().lower()

    for epoch in range(int(n_epochs)):
        epoch_start = time.time()
        epoch_losses = []

        for batch_X, batch_Y in loader:
            losses = wgan.train_step(batch_Y, batch_X, noise_dim)
            epoch_losses.append(losses)

        avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0].keys()}
        wgan.training_history.append(avg_losses)
        epoch_time = time.time() - epoch_start
        avg_time_per_epoch = (time.time() - start_time) / float(epoch + 1)

        if (epoch + 1) % int(print_every) == 0:
            print(
                f"[{label}] Epoch {epoch+1}/{n_epochs} - "
                f"W-dist: {avg_losses['wasserstein_distance']:.4f}, "
                f"C Loss: {avg_losses['c_loss']:.4f}, "
                f"G Loss: {avg_losses['g_loss']:.4f}, "
                f"Epoch Time: {epoch_time:.4f}s, "
                f"Avg/Epoch: {avg_time_per_epoch:.4f}s"
            )

        do_checkpoint = (
            ((epoch + 1) % int(checkpoint_every) == 0)
            or (epoch == 0)
            or (epoch + 1 == int(n_epochs))
        )
        if do_checkpoint:
            current_w1 = _arm_checkpoint_w1_score_worker(
                wgan=wgan,
                X_t=X_t,
                Y_t_scaled=Y_t,
                noise_dim=noise_dim,
                n_draws=checkpoint_n_draws,
                seed=10_000 + int(seed_offset) + int(epoch + 1) * 100,
            )
            checkpoint_log.append(
                {
                    "epoch": int(epoch + 1),
                    "mean_w1_scaled": float(current_w1),
                }
            )
            print(
                f"[{label}] checkpoint epoch {epoch+1}/{n_epochs} - "
                f"mean exact W1 (scaled Y, {checkpoint_n_draws} draws): {current_w1:.6f}"
            )
            if current_w1 < best_w1:
                best_w1 = float(current_w1)
                best_epoch = int(epoch + 1)
                best_generator_state = _clone_state_dict_cpu(wgan.generator)
                best_critic_state = _clone_state_dict_cpu(wgan.critic)
                print(
                    f"[{label}] new best checkpoint at epoch {epoch+1} - "
                    f"mean exact W1 (scaled Y): {current_w1:.6f}"
                )

    if best_generator_state is not None:
        wgan.generator.load_state_dict(best_generator_state)
        wgan.critic.load_state_dict(best_critic_state)

    elapsed = time.time() - start_time
    print(
        f"[{label}] worker finished | best_epoch={best_epoch} | "
        f"best_exact_w1={best_w1:.6f} | elapsed={elapsed:.2f}s",
        flush=True,
    )

    if best_generator_state is None or best_critic_state is None:
        raise RuntimeError(f"[{label}] no best checkpoint was recorded")

    generator_path = save_dir / f"{label_slug}_best_generator.pt"
    critic_path = save_dir / f"{label_slug}_best_critic.pt"
    history_path = save_dir / f"{label_slug}_history.pt"
    checkpoint_log_path = save_dir / f"{label_slug}_checkpoint_log.json"

    torch.save(best_generator_state, generator_path)
    torch.save(best_critic_state, critic_path)
    torch.save(wgan.training_history, history_path)
    with open(checkpoint_log_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_log, f, indent=2)

    queue.put(
        {
            "label": label,
            "generator_path": str(generator_path),
            "critic_path": str(critic_path),
            "history_path": str(history_path),
            "checkpoint_log_path": str(checkpoint_log_path),
            "best_epoch": int(best_epoch),
            "best_w1": float(best_w1),
        }
    )


class Generator(nn.Module):
    def __init__(self, covariate_dim, noise_dim=10, hidden_dim=128, n_layers=3):
        super().__init__()
        input_dim = noise_dim + covariate_dim

        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, noise, covariates):
        if covariates.ndim != 2:
            covariates = covariates.view(covariates.size(0), -1)
        x = torch.cat([noise, covariates], dim=1)
        x = self.hidden(x)
        return self.output(x)


class Critic(nn.Module):
    def __init__(self, covariate_dim, hidden_dim=256, n_layers=4):
        super().__init__()
        input_dim = 1 + covariate_dim

        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, outcome, covariates):
        if outcome.ndim == 1:
            outcome = outcome.view(-1, 1)
        elif outcome.ndim == 3 and outcome.shape[1] == 1 and outcome.shape[2] == 1:
            outcome = outcome.view(-1, 1)

        if covariates.ndim != 2:
            covariates = covariates.view(covariates.size(0), -1)

        inp = torch.cat([outcome, covariates], dim=1)
        return self.network(inp)


class ProjectionCritic(nn.Module):
    def __init__(self, covariate_dim, hidden_dim=256, n_layers=3):
        super().__init__()

        layers = []
        in_dim = 1
        for _ in range(n_layers):
            layers.append(spectral_norm(nn.Linear(in_dim, hidden_dim)))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = hidden_dim
        self.y_net = nn.Sequential(*layers)

        self.x_embed = spectral_norm(nn.Linear(covariate_dim, hidden_dim))
        self.head = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, outcome, covariates):
        if outcome.ndim == 1:
            outcome = outcome.view(-1, 1)
        elif outcome.ndim == 3 and outcome.shape[1] == 1 and outcome.shape[2] == 1:
            outcome = outcome.view(-1, 1)

        if covariates.ndim != 2:
            covariates = covariates.view(covariates.size(0), -1)

        fy = self.y_net(outcome)
        ux = self.x_embed(covariates)
        proj = (fy * ux).sum(dim=1, keepdim=True)

        return self.head(fy) + proj


class CausalWGAN:
    def __init__(self, generator, critic, lambda_gp=10, n_critic=5, g_lr=5e-5, c_lr=2e-4, device=None):
        self.generator = generator
        self.critic = critic
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.0, 0.9),
        )
        self.c_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=c_lr,
            betas=(0.0, 0.9),
        )

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator.to(self.device)
        self.critic.to(self.device)

        self.training_history = []

    def set_device(self, device):
        self.device = torch.device(device)
        self.generator.to(self.device)
        self.critic.to(self.device)

    def gradient_penalty(self, real_outcome, fake_outcome, covariates):
        batch_size = real_outcome.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)

        interpolates = alpha * real_outcome + (1 - alpha) * fake_outcome
        interpolates.requires_grad_(True)

        critic_interpolates = self.critic(interpolates, covariates)

        gradients = grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()

    def train_step(self, real_outcome, covariates, noise_dim=10):
        batch_size = real_outcome.size(0)

        if real_outcome.ndim == 1:
            real_outcome = real_outcome.view(-1, 1)
        elif real_outcome.ndim == 3 and real_outcome.shape[1] == 1 and real_outcome.shape[2] == 1:
            real_outcome = real_outcome.view(-1, 1)

        if covariates.ndim != 2:
            covariates = covariates.view(covariates.size(0), -1)

        real_outcome = real_outcome.to(self.device, non_blocking=True)
        covariates = covariates.to(self.device, non_blocking=True)

        for _ in range(self.n_critic):
            self.c_optimizer.zero_grad(set_to_none=True)

            noise = torch.randn(batch_size, noise_dim, device=self.device)
            fake_outcome = self.generator(noise, covariates)

            real_score = self.critic(real_outcome, covariates)
            fake_score = self.critic(fake_outcome.detach(), covariates)

            gp = self.gradient_penalty(real_outcome, fake_outcome.detach(), covariates)
            c_loss = fake_score.mean() - real_score.mean() + self.lambda_gp * gp

            c_loss.backward()
            self.c_optimizer.step()

        self.g_optimizer.zero_grad(set_to_none=True)

        noise = torch.randn(batch_size, noise_dim, device=self.device)
        fake_outcome = self.generator(noise, covariates)
        fake_score = self.critic(fake_outcome, covariates)

        g_loss = -fake_score.mean()
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "c_loss": c_loss.item(),
            "g_loss": g_loss.item(),
            "real_score": real_score.mean().item(),
            "fake_score": fake_score.mean().item(),
            "gp": gp.item(),
            "wasserstein_distance": (real_score.mean() - fake_score.mean()).item(),
        }

    def generate(self, covariates, noise_dim=10, n_samples=1):
        self.generator.eval()

        if covariates.ndim != 2:
            covariates = covariates.view(covariates.size(0), -1)

        with torch.no_grad():
            n_obs = covariates.size(0)
            covariates = covariates.to(self.device)

            if n_samples == 1:
                noise = torch.randn(n_obs, noise_dim, device=self.device)
                outcomes = self.generator(noise, covariates)
            else:
                outs = []
                for _ in range(n_samples):
                    noise = torch.randn(n_obs, noise_dim, device=self.device)
                    outs.append(self.generator(noise, covariates))
                outcomes = torch.cat(outs, dim=1)

        self.generator.train()
        return outcomes.cpu()


class WGANmodel:
    """
    Two WGANs (control/treated). Uses sklearn scalers -> normalization/denorm are NumPy,
    but inputs can be torch tensors (CPU or GPU).
    """

    def __init__(
        self,
        covariate_dim,
        noise_dim=10,
        hidden_dim=128,
        n_layers=3,
        lambda_gp=10,
        n_critic=5,
        g_lr=5e-5,
        c_lr=2e-4,
        target_ate=None,
        lambda_ate=1.0,
        initialization_seed: int | None = 42,
    ):
        # The network weights are initialized by the constructors below, not
        # by ``fit``.  Seed before constructing them so a fit is reproducible
        # from its recorded initialization and training seeds.
        self.initialization_seed = (
            None if initialization_seed is None else int(initialization_seed)
        )
        self.training_seed = None
        if self.initialization_seed is not None:
            torch.manual_seed(self.initialization_seed)
            np.random.seed(self.initialization_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.initialization_seed)

        gen_0 = Generator(covariate_dim, noise_dim, hidden_dim, n_layers)
        critic_0 = Critic(covariate_dim, hidden_dim=hidden_dim, n_layers=n_layers)
        self.wgan_control = CausalWGAN(gen_0, critic_0, lambda_gp, n_critic, g_lr=g_lr, c_lr=c_lr)

        gen_1 = Generator(covariate_dim, noise_dim, hidden_dim, n_layers)
        critic_1 = Critic(covariate_dim, hidden_dim=hidden_dim, n_layers=n_layers)
        self.wgan_treated = CausalWGAN(gen_1, critic_1, lambda_gp, n_critic, g_lr=g_lr, c_lr=c_lr)

        self.covariate_scaler = StandardScaler()
        self.outcome_scaler_0 = StandardScaler()
        self.outcome_scaler_1 = StandardScaler()

        self.noise_dim = noise_dim
        self.target_ate = target_ate
        self.lambda_ate = lambda_ate
        self.target_ate_normalized = None
        self.checkpoint_log_control = []
        self.checkpoint_log_treated = []
        self.best_epoch_control = None
        self.best_epoch_treated = None
        self.best_w1_scaled_control = None
        self.best_w1_scaled_treated = None

    def sample_y0(self, X, n_samples=1, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = X.detach().cpu().numpy()

        X_np = np.asarray(X_np).reshape(len(X_np), -1)
        X_norm = self.covariate_scaler.transform(X_np)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        y_0_samples_t = self.wgan_control.generate(X_tensor, self.noise_dim, n_samples)
        y_0_samples = self.outcome_scaler_0.inverse_transform(
            y_0_samples_t.detach().cpu().numpy()
        )

        if n_samples == 1:
            return y_0_samples.reshape(-1)
        return y_0_samples

    def sample_y1(self, X, n_samples=1, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if isinstance(X, np.ndarray):
            X_np = X
        else:
            X_np = X.detach().cpu().numpy()

        X_np = np.asarray(X_np).reshape(len(X_np), -1)
        X_norm = self.covariate_scaler.transform(X_np)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        y_1_samples_t = self.wgan_treated.generate(X_tensor, self.noise_dim, n_samples)
        y_1_samples = self.outcome_scaler_1.inverse_transform(
            y_1_samples_t.detach().cpu().numpy()
        )

        if n_samples == 1:
            return y_1_samples.reshape(-1)
        return y_1_samples

    def ate_penalty(self, y_0_fake, y_1_fake):
        ate_current = y_1_fake.mean() - y_0_fake.mean()
        target = self.target_ate_normalized if self.target_ate_normalized is not None else 0.0
        penalty = (ate_current - target) ** 2
        return penalty, ate_current.item()

    def _arm_checkpoint_w1_score(self, wgan, X_t, Y_t_scaled, *, n_draws: int, seed: int | None = None):
        """
        Mean exact 1D Wasserstein distance on the arm-specific scaled outcome,
        averaged across repeated generator draws for stability.
        """
        y_real = Y_t_scaled.detach().cpu().numpy().reshape(-1)
        scores = []

        for draw_idx in range(int(n_draws)):
            if seed is not None:
                local_seed = int(seed) + int(draw_idx)
                torch.manual_seed(local_seed)
                np.random.seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)

            y_fake = wgan.generate(
                X_t,
                noise_dim=self.noise_dim,
                n_samples=1,
            ).detach().cpu().numpy().reshape(-1)

            scores.append(float(wasserstein_distance(y_real, y_fake)))

        return float(np.mean(scores))

    @staticmethod
    def _clone_state_dict_cpu(module: nn.Module):
        return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

    def _parallel_training_plan(self):
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        n_cpus = available_cpus()

        if self.target_ate is not None:
            if n_gpus >= 1:
                return {
                    "mode": "sequential",
                    "control_device": "cuda:0",
                    "treated_device": "cuda:0",
                    "control_threads": 1,
                    "treated_threads": 1,
                    "reason": "ATE-constrained training must remain joint/sequential",
                }
            return {
                "mode": "sequential",
                "control_device": "cpu",
                "treated_device": "cpu",
                "control_threads": n_cpus,
                "treated_threads": n_cpus,
                "reason": "ATE-constrained training must remain joint/sequential",
            }

        if n_gpus >= 2:
            cpu_half = max(1, n_cpus // 2)
            return {
                "mode": "parallel",
                "control_device": "cuda:0",
                "treated_device": "cuda:1",
                "control_threads": cpu_half,
                "treated_threads": max(1, n_cpus - cpu_half),
                "reason": "Two GPUs available: split control/treated across devices",
            }

        if n_gpus == 1:
            return {
                "mode": "sequential",
                "control_device": "cuda:0",
                "treated_device": "cuda:0",
                "control_threads": max(1, n_cpus),
                "treated_threads": max(1, n_cpus),
                "reason": "Single GPU available: sequential arm training on cuda:0",
            }

        if n_cpus >= 2:
            cpu_half = max(1, n_cpus // 2)
            return {
                "mode": "parallel",
                "control_device": "cpu",
                "treated_device": "cpu",
                "control_threads": cpu_half,
                "treated_threads": max(1, n_cpus - cpu_half),
                "reason": "CPU-only environment with multiple cores: split arms across processes",
            }

        return {
            "mode": "sequential",
            "control_device": "cpu",
            "treated_device": "cpu",
            "control_threads": 1,
            "treated_threads": 1,
            "reason": "Single CPU core available",
        }

    def _run_parallel_unconstrained_training(
        self,
        *,
        X_0_tensor,
        Y_0_tensor,
        X_1_tensor,
        Y_1_tensor,
        batch_size,
        n_epochs,
        print_every,
        checkpoint_every,
        checkpoint_n_draws,
        control_device,
        treated_device,
        control_threads,
        treated_threads,
        seed: int,
    ):
        start_method = "spawn"
        print(f"WGAN parallel start method: {start_method}")
        ctx = mp.get_context(start_method)
        q0 = ctx.Queue()
        q1 = ctx.Queue()
        worker_root = Path(__file__).resolve().parent / "mp_worker_checkpoints"
        worker_root.mkdir(parents=True, exist_ok=True)
        # Each fit call gets a private checkpoint directory so concurrent jobs
        # cannot overwrite one another's worker artifacts on the shared FS.
        worker_dir = Path(
            tempfile.mkdtemp(
                prefix=f"wgan_mp_seed{int(seed)}_pid{os.getpid()}_",
                dir=str(worker_root),
            )
        )

        p0 = ctx.Process(
            target=_train_single_arm_worker,
            kwargs={
                "wgan": self.wgan_control,
                "X_t": X_0_tensor,
                "Y_t": Y_0_tensor,
                "noise_dim": self.noise_dim,
                "n_epochs": n_epochs,
                "print_every": print_every,
                "batch_size": batch_size,
                "label": "Control",
                "checkpoint_every": checkpoint_every,
                "checkpoint_n_draws": checkpoint_n_draws,
                "worker_threads": control_threads,
                "device": control_device,
                "base_seed": int(seed),
                "seed_offset": 0,
                "queue": q0,
                "save_dir": str(worker_dir),
            },
        )
        p1 = ctx.Process(
            target=_train_single_arm_worker,
            kwargs={
                "wgan": self.wgan_treated,
                "X_t": X_1_tensor,
                "Y_t": Y_1_tensor,
                "noise_dim": self.noise_dim,
                "n_epochs": n_epochs,
                "print_every": print_every,
                "batch_size": batch_size,
                "label": "Treated",
                "checkpoint_every": checkpoint_every,
                "checkpoint_n_draws": checkpoint_n_draws,
                "worker_threads": treated_threads,
                "device": treated_device,
                "base_seed": int(seed) + 500_000,
                "seed_offset": 500_000,
                "queue": q1,
                "save_dir": str(worker_dir),
            },
        )

        try:
            p0.start()
            p1.start()

            res0 = q0.get()
            res1 = q1.get()

            p0.join()
            p1.join()

            if p0.exitcode != 0:
                raise RuntimeError(f"Control WGAN worker failed with exit code {p0.exitcode}")
            if p1.exitcode != 0:
                raise RuntimeError(f"Treated WGAN worker failed with exit code {p1.exitcode}")

            self.wgan_control.set_device(control_device)
            self.wgan_treated.set_device(treated_device)
            self.wgan_control.generator.load_state_dict(torch.load(res0["generator_path"], map_location="cpu"))
            self.wgan_control.critic.load_state_dict(torch.load(res0["critic_path"], map_location="cpu"))
            self.wgan_treated.generator.load_state_dict(torch.load(res1["generator_path"], map_location="cpu"))
            self.wgan_treated.critic.load_state_dict(torch.load(res1["critic_path"], map_location="cpu"))
            self.wgan_control.training_history = list(torch.load(res0["history_path"], map_location="cpu"))
            self.wgan_treated.training_history = list(torch.load(res1["history_path"], map_location="cpu"))
            with open(res0["checkpoint_log_path"], "r", encoding="utf-8") as f:
                self.checkpoint_log_control = list(json.load(f))
            with open(res1["checkpoint_log_path"], "r", encoding="utf-8") as f:
                self.checkpoint_log_treated = list(json.load(f))
            self.best_epoch_control = res0["best_epoch"]
            self.best_w1_scaled_control = res0["best_w1"]
            self.best_epoch_treated = res1["best_epoch"]
            self.best_w1_scaled_treated = res1["best_w1"]
        finally:
            shutil.rmtree(worker_dir, ignore_errors=True)

    def _maybe_record_best_checkpoint(
        self,
        *,
        wgan,
        X_t,
        Y_t_scaled,
        epoch: int,
        checkpoint_every: int,
        checkpoint_n_draws: int,
        label: str,
        checkpoint_log: list,
        best_state: dict,
        seed_offset: int = 0,
    ):
        do_checkpoint = (
            ((epoch + 1) % int(checkpoint_every) == 0)
            or (epoch == 0)
            or (epoch + 1 == int(best_state["n_epochs"]))
        )
        if not do_checkpoint:
            return

        current_w1 = self._arm_checkpoint_w1_score(
            wgan=wgan,
            X_t=X_t,
            Y_t_scaled=Y_t_scaled,
            n_draws=checkpoint_n_draws,
            seed=10_000 + int(seed_offset) + int(epoch + 1) * 100,
        )

        checkpoint_log.append(
            {
                "epoch": int(epoch + 1),
                "mean_w1_scaled": float(current_w1),
            }
        )

        print(
            f"[{label}] checkpoint epoch {epoch+1}/{best_state['n_epochs']} - "
            f"mean exact W1 (scaled Y, {checkpoint_n_draws} draws): {current_w1:.6f}"
        )

        if current_w1 < best_state["best_w1"]:
            best_state["best_w1"] = float(current_w1)
            best_state["best_epoch"] = int(epoch + 1)
            best_state["best_generator_state"] = self._clone_state_dict_cpu(wgan.generator)
            best_state["best_critic_state"] = self._clone_state_dict_cpu(wgan.critic)

            print(
                f"[{label}] new best checkpoint at epoch {epoch+1} - "
                f"mean exact W1 (scaled Y): {current_w1:.6f}"
            )

    def fit(
        self,
        X,
        A,
        Y,
        batch_size=512,
        n_epochs=5000,
        print_every=500,
        resume=False,
        checkpoint_every=200,
        checkpoint_n_draws=5,
        seed=42,
    ):
        self.training_seed = int(seed)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(A):
            A = A.detach().cpu().numpy()
        if torch.is_tensor(Y):
            Y = Y.detach().cpu().numpy()

        X = np.asarray(X).reshape(len(X), -1)
        A = np.asarray(A).reshape(-1)
        Y = np.asarray(Y).reshape(-1)

        mask_control = A == 0
        mask_treated = A == 1

        X_0, Y_0 = X[mask_control], Y[mask_control]
        X_1, Y_1 = X[mask_treated], Y[mask_treated]

        print(f"Control group: {len(Y_0)} samples")
        print(f"Treated group: {len(Y_1)} samples")
        print(f"Batch size: {int(batch_size)}")

        self.checkpoint_log_control = []
        self.checkpoint_log_treated = []
        self.best_epoch_control = None
        self.best_epoch_treated = None
        self.best_w1_scaled_control = None
        self.best_w1_scaled_treated = None

        if not resume:
            self.covariate_scaler.fit(X)
            self.outcome_scaler_0.fit(Y_0.reshape(-1, 1))
            self.outcome_scaler_1.fit(Y_1.reshape(-1, 1))

        X_0_norm = self.covariate_scaler.transform(X_0)
        X_1_norm = self.covariate_scaler.transform(X_1)

        Y_0_norm = self.outcome_scaler_0.transform(Y_0.reshape(-1, 1)).reshape(-1)
        Y_1_norm = self.outcome_scaler_1.transform(Y_1.reshape(-1, 1)).reshape(-1)

        if self.target_ate is not None:
            scale_factor = (self.outcome_scaler_1.scale_[0] + self.outcome_scaler_0.scale_[0]) / 2.0
            self.target_ate_normalized = float(self.target_ate) / float(scale_factor)
            print(f"\nTarget ATE (original outcome scale): {self.target_ate:.4f}")
            print(f"Target ATE (normalized training scale): {self.target_ate_normalized:.4f}")
        else:
            self.target_ate_normalized = None

        X_0_tensor = torch.tensor(X_0_norm, dtype=torch.float32)
        Y_0_tensor = torch.tensor(Y_0_norm, dtype=torch.float32).view(-1, 1)

        X_1_tensor = torch.tensor(X_1_norm, dtype=torch.float32)
        Y_1_tensor = torch.tensor(Y_1_norm, dtype=torch.float32).view(-1, 1)

        plan = self._parallel_training_plan()
        rt = runtime_diagnostics()
        print("\nWGAN runtime diagnostics:")
        print(f"  Available CPUs:     {rt['available_cpus']}")
        print(f"  CPU affinity:       {rt['cpu_affinity'] if rt['cpu_affinity'] is not None else 'unavailable'}")
        print(f"  CUDA available:     {rt['cuda_available']}")
        print(f"  CUDA device count:  {rt['cuda_device_count']}")
        print(f"  CUDA_VISIBLE_DEVICES: {rt['cuda_visible_devices']}")
        print("\nWGAN resource plan:")
        print(f"  Mode:            {plan['mode']}")
        print(f"  Control device:  {plan['control_device']}")
        print(f"  Treated device:  {plan['treated_device']}")
        print(f"  Control threads: {plan['control_threads']}")
        print(f"  Treated threads: {plan['treated_threads']}")
        print(f"  Reason:          {plan['reason']}")

        if self.target_ate is not None:
            set_worker_threads(plan["control_threads"])
            self.wgan_control.set_device(plan["control_device"])
            self.wgan_treated.set_device(plan["treated_device"])
            print("\nTraining WGANs with ATE constraint...")
            self._train_wgan_constrained(
                X_0_tensor,
                Y_0_tensor,
                X_1_tensor,
                Y_1_tensor,
                batch_size,
                n_epochs,
                print_every,
                checkpoint_every,
                checkpoint_n_draws,
            )
        else:
            if plan["mode"] == "parallel":
                print("\nTraining control and treated WGANs in parallel...")
                self._run_parallel_unconstrained_training(
                    X_0_tensor=X_0_tensor,
                    Y_0_tensor=Y_0_tensor,
                    X_1_tensor=X_1_tensor,
                    Y_1_tensor=Y_1_tensor,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    print_every=print_every,
                    checkpoint_every=checkpoint_every,
                    checkpoint_n_draws=checkpoint_n_draws,
                    control_device=plan["control_device"],
                    treated_device=plan["treated_device"],
                    control_threads=plan["control_threads"],
                    treated_threads=plan["treated_threads"],
                    seed=int(seed),
                )
            else:
                set_worker_threads(plan["control_threads"])
                self.wgan_control.set_device(plan["control_device"])
                self.wgan_treated.set_device(plan["treated_device"])

                print("\nTraining WGAN for control group (A=0)...")
                print(f"  device={plan['control_device']} | threads={plan['control_threads']} | n={X_0_tensor.size(0)}")
                self._train_wgan(
                    self.wgan_control,
                    X_0_tensor,
                    Y_0_tensor,
                    batch_size,
                    n_epochs,
                    print_every,
                    checkpoint_every,
                    checkpoint_n_draws,
                    "Control",
                    self.checkpoint_log_control,
                    seed_offset=0,
                )

                print("\nTraining WGAN for treated group (A=1)...")
                print(f"  device={plan['treated_device']} | threads={plan['treated_threads']} | n={X_1_tensor.size(0)}")
                self._train_wgan(
                    self.wgan_treated,
                    X_1_tensor,
                    Y_1_tensor,
                    batch_size,
                    n_epochs,
                    print_every,
                    checkpoint_every,
                    checkpoint_n_draws,
                    "Treated",
                    self.checkpoint_log_treated,
                    seed_offset=500_000,
                )

            print("\nTraining complete!")

    def _train_wgan(
        self,
        wgan,
        X,
        Y,
        batch_size,
        n_epochs,
        print_every,
        checkpoint_every,
        checkpoint_n_draws,
        label,
        checkpoint_log,
        seed_offset=0,
    ):
        start_time = time.time()
        dataset = TensorDataset(X, Y)
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            pin_memory=True,
        )
        best_state = {
            "best_w1": float("inf"),
            "best_epoch": None,
            "best_generator_state": None,
            "best_critic_state": None,
            "n_epochs": int(n_epochs),
        }

        for epoch in range(n_epochs):
            epoch_start = time.time()
            epoch_losses = []

            for batch_X, batch_Y in loader:
                losses = wgan.train_step(batch_Y, batch_X, self.noise_dim)
                epoch_losses.append(losses)

            avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0].keys()}
            wgan.training_history.append(avg_losses)
            epoch_time = time.time() - epoch_start
            avg_time_per_epoch = (time.time() - start_time) / float(epoch + 1)

            if (epoch + 1) % print_every == 0:
                print(
                    f"[{label}] Epoch {epoch+1}/{n_epochs} - "
                    f"W-dist: {avg_losses['wasserstein_distance']:.4f}, "
                    f"C Loss: {avg_losses['c_loss']:.4f}, "
                    f"G Loss: {avg_losses['g_loss']:.4f}, "
                    f"Epoch Time: {epoch_time:.4f}s, "
                    f"Avg/Epoch: {avg_time_per_epoch:.4f}s"
                )

            self._maybe_record_best_checkpoint(
                wgan=wgan,
                X_t=X,
                Y_t_scaled=Y,
                epoch=epoch,
                checkpoint_every=checkpoint_every,
                checkpoint_n_draws=checkpoint_n_draws,
                label=label,
                checkpoint_log=checkpoint_log,
                best_state=best_state,
                seed_offset=seed_offset,
            )

        if best_state["best_generator_state"] is not None:
            wgan.generator.load_state_dict(best_state["best_generator_state"])
            wgan.critic.load_state_dict(best_state["best_critic_state"])

        if label.lower().startswith("control"):
            self.best_epoch_control = best_state["best_epoch"]
            self.best_w1_scaled_control = best_state["best_w1"]
        else:
            self.best_epoch_treated = best_state["best_epoch"]
            self.best_w1_scaled_treated = best_state["best_w1"]

        elapsed = time.time() - start_time
        print(
            f"[{label}] fit finished | best_epoch={best_state['best_epoch']} | "
            f"best_exact_w1={best_state['best_w1']:.6f} | elapsed={elapsed:.2f}s"
        )

    def _train_wgan_constrained(
        self,
        X_0,
        Y_0,
        X_1,
        Y_1,
        batch_size,
        n_epochs,
        print_every,
        checkpoint_every,
        checkpoint_n_draws,
    ):
        n_samples_0 = X_0.size(0)
        n_samples_1 = X_1.size(0)
        start_time = time.time()
        device = self.wgan_control.device
        best_state_0 = {
            "best_w1": float("inf"),
            "best_epoch": None,
            "best_generator_state": None,
            "best_critic_state": None,
            "n_epochs": int(n_epochs),
        }
        best_state_1 = {
            "best_w1": float("inf"),
            "best_epoch": None,
            "best_generator_state": None,
            "best_critic_state": None,
            "n_epochs": int(n_epochs),
        }

        for epoch in range(n_epochs):
            epoch_start = time.time()
            indices_0 = torch.randperm(n_samples_0)
            indices_1 = torch.randperm(n_samples_1)

            epoch_losses = {
                "c_loss_0": [],
                "g_loss_0": [],
                "wdist_0": [],
                "gp_0": [],
                "c_loss_1": [],
                "g_loss_1": [],
                "wdist_1": [],
                "gp_1": [],
                "ate_penalty": [],
                "ate_current": [],
            }

            max_batches = max(
                (n_samples_0 + batch_size - 1) // batch_size,
                (n_samples_1 + batch_size - 1) // batch_size,
            )

            for batch_idx in range(max_batches):
                start_0 = (batch_idx * batch_size) % n_samples_0
                end_0 = min(start_0 + batch_size, n_samples_0)
                batch_indices_0 = indices_0[start_0:end_0]

                start_1 = (batch_idx * batch_size) % n_samples_1
                end_1 = min(start_1 + batch_size, n_samples_1)
                batch_indices_1 = indices_1[start_1:end_1]

                batch_X_0 = X_0[batch_indices_0].to(device)
                batch_Y_0 = Y_0[batch_indices_0].to(device)

                batch_X_1 = X_1[batch_indices_1].to(device)
                batch_Y_1 = Y_1[batch_indices_1].to(device)

                for _ in range(self.wgan_control.n_critic):
                    self.wgan_control.c_optimizer.zero_grad()
                    noise = torch.randn(batch_X_0.size(0), self.noise_dim, device=device)
                    fake_Y_0 = self.wgan_control.generator(noise, batch_X_0)

                    real_score_0 = self.wgan_control.critic(batch_Y_0, batch_X_0)
                    fake_score_0 = self.wgan_control.critic(fake_Y_0.detach(), batch_X_0)

                    gp_0 = self.wgan_control.gradient_penalty(batch_Y_0, fake_Y_0.detach(), batch_X_0)
                    c_loss_0 = fake_score_0.mean() - real_score_0.mean() + self.wgan_control.lambda_gp * gp_0

                    c_loss_0.backward()
                    self.wgan_control.c_optimizer.step()

                for _ in range(self.wgan_treated.n_critic):
                    self.wgan_treated.c_optimizer.zero_grad()
                    noise = torch.randn(batch_X_1.size(0), self.noise_dim, device=device)
                    fake_Y_1 = self.wgan_treated.generator(noise, batch_X_1)

                    real_score_1 = self.wgan_treated.critic(batch_Y_1, batch_X_1)
                    fake_score_1 = self.wgan_treated.critic(fake_Y_1.detach(), batch_X_1)

                    gp_1 = self.wgan_treated.gradient_penalty(batch_Y_1, fake_Y_1.detach(), batch_X_1)
                    c_loss_1 = fake_score_1.mean() - real_score_1.mean() + self.wgan_treated.lambda_gp * gp_1

                    c_loss_1.backward()
                    self.wgan_treated.c_optimizer.step()

                self.wgan_control.g_optimizer.zero_grad()
                self.wgan_treated.g_optimizer.zero_grad()

                n_ate_samples = min(batch_X_0.size(0), batch_X_1.size(0))
                X_ate = torch.cat([batch_X_0[:n_ate_samples], batch_X_1[:n_ate_samples]], dim=0)

                noise_0 = torch.randn(X_ate.size(0), self.noise_dim, device=device)
                noise_1 = torch.randn(X_ate.size(0), self.noise_dim, device=device)

                fake_Y_0_ate = self.wgan_control.generator(noise_0, X_ate)
                fake_Y_1_ate = self.wgan_treated.generator(noise_1, X_ate)

                fake_score_0_gen = self.wgan_control.critic(
                    fake_Y_0_ate[:n_ate_samples],
                    X_ate[:n_ate_samples],
                )
                fake_score_1_gen = self.wgan_treated.critic(
                    fake_Y_1_ate[:n_ate_samples],
                    X_ate[:n_ate_samples],
                )

                g_loss_0 = -fake_score_0_gen.mean()
                g_loss_1 = -fake_score_1_gen.mean()

                ate_pen, ate_curr = self.ate_penalty(fake_Y_0_ate, fake_Y_1_ate)

                total_g_loss_0 = g_loss_0 + self.lambda_ate * ate_pen
                total_g_loss_1 = g_loss_1 + self.lambda_ate * ate_pen

                total_g_loss_0.backward(retain_graph=True)
                total_g_loss_1.backward()

                self.wgan_control.g_optimizer.step()
                self.wgan_treated.g_optimizer.step()

                epoch_losses["c_loss_0"].append(c_loss_0.item())
                epoch_losses["g_loss_0"].append(g_loss_0.item())
                epoch_losses["wdist_0"].append((real_score_0.mean() - fake_score_0.mean()).item())
                epoch_losses["gp_0"].append(gp_0.item())

                epoch_losses["c_loss_1"].append(c_loss_1.item())
                epoch_losses["g_loss_1"].append(g_loss_1.item())
                epoch_losses["wdist_1"].append((real_score_1.mean() - fake_score_1.mean()).item())
                epoch_losses["gp_1"].append(gp_1.item())

                epoch_losses["ate_penalty"].append(ate_pen.item())
                epoch_losses["ate_current"].append(ate_curr)

            avg_losses_0 = {
                "c_loss": np.mean(epoch_losses["c_loss_0"]),
                "g_loss": np.mean(epoch_losses["g_loss_0"]),
                "wasserstein_distance": np.mean(epoch_losses["wdist_0"]),
                "gp": np.mean(epoch_losses["gp_0"]),
                "ate_penalty": np.mean(epoch_losses["ate_penalty"]),
                "ate_current": np.mean(epoch_losses["ate_current"]),
            }
            avg_losses_1 = {
                "c_loss": np.mean(epoch_losses["c_loss_1"]),
                "g_loss": np.mean(epoch_losses["g_loss_1"]),
                "wasserstein_distance": np.mean(epoch_losses["wdist_1"]),
                "gp": np.mean(epoch_losses["gp_1"]),
                "ate_penalty": np.mean(epoch_losses["ate_penalty"]),
                "ate_current": np.mean(epoch_losses["ate_current"]),
            }

            self.wgan_control.training_history.append(avg_losses_0)
            self.wgan_treated.training_history.append(avg_losses_1)
            epoch_time = time.time() - epoch_start
            avg_time_per_epoch = (time.time() - start_time) / float(epoch + 1)

            if (epoch + 1) % print_every == 0:
                scale_factor = (self.outcome_scaler_1.scale_[0] + self.outcome_scaler_0.scale_[0]) / 2.0
                ate_original = avg_losses_0["ate_current"] * scale_factor
                tgt = self.target_ate if self.target_ate is not None else 0.0
                print(f"Epoch {epoch+1}/{n_epochs}")
                print(
                    f"  [Control]  W-dist: {avg_losses_0['wasserstein_distance']:.4f}, "
                    f"G Loss: {avg_losses_0['g_loss']:.4f}"
                )
                print(
                    f"  [Treated]  W-dist: {avg_losses_1['wasserstein_distance']:.4f}, "
                    f"G Loss: {avg_losses_1['g_loss']:.4f}"
                )
                print(
                    f"  [ATE]      Target: {tgt:.4f}, "
                    f"Current: {ate_original:.4f}, "
                    f"Penalty: {avg_losses_0['ate_penalty']:.6f}"
                )
                print(
                    f"  [Timing]   Epoch Time: {epoch_time:.4f}s, "
                    f"Avg/Epoch: {avg_time_per_epoch:.4f}s"
                )

            self._maybe_record_best_checkpoint(
                wgan=self.wgan_control,
                X_t=X_0,
                Y_t_scaled=Y_0,
                epoch=epoch,
                checkpoint_every=checkpoint_every,
                checkpoint_n_draws=checkpoint_n_draws,
                label="Control",
                checkpoint_log=self.checkpoint_log_control,
                best_state=best_state_0,
                seed_offset=0,
            )
            self._maybe_record_best_checkpoint(
                wgan=self.wgan_treated,
                X_t=X_1,
                Y_t_scaled=Y_1,
                epoch=epoch,
                checkpoint_every=checkpoint_every,
                checkpoint_n_draws=checkpoint_n_draws,
                label="Treated",
                checkpoint_log=self.checkpoint_log_treated,
                best_state=best_state_1,
                seed_offset=500_000,
            )

        if best_state_0["best_generator_state"] is not None:
            self.wgan_control.generator.load_state_dict(best_state_0["best_generator_state"])
            self.wgan_control.critic.load_state_dict(best_state_0["best_critic_state"])
        if best_state_1["best_generator_state"] is not None:
            self.wgan_treated.generator.load_state_dict(best_state_1["best_generator_state"])
            self.wgan_treated.critic.load_state_dict(best_state_1["best_critic_state"])

        self.best_epoch_control = best_state_0["best_epoch"]
        self.best_w1_scaled_control = best_state_0["best_w1"]
        self.best_epoch_treated = best_state_1["best_epoch"]
        self.best_w1_scaled_treated = best_state_1["best_w1"]
        elapsed = time.time() - start_time
        print(
            f"Constrained WGAN fit finished | control_best_epoch={self.best_epoch_control} | "
            f"treated_best_epoch={self.best_epoch_treated} | elapsed={elapsed:.2f}s"
        )

    def estimate_cate(self, X, n_samples=100):
        Y_0_samples = self.sample_y0(X, n_samples=n_samples)
        Y_1_samples = self.sample_y1(X, n_samples=n_samples)

        if n_samples == 1:
            cate = (Y_1_samples - Y_0_samples).reshape(-1)
            cate_se = None
            y0_mean = Y_0_samples.reshape(-1)
            y1_mean = Y_1_samples.reshape(-1)
        else:
            te_samples = Y_1_samples - Y_0_samples
            cate = te_samples.mean(axis=1)
            cate_se = te_samples.std(axis=1)
            y0_mean = Y_0_samples.mean(axis=1)
            y1_mean = Y_1_samples.mean(axis=1)

        return {
            "cate": cate,
            "cate_se": cate_se,
            "y0_mean": y0_mean,
            "y1_mean": y1_mean,
            "y0_samples": Y_0_samples,
            "y1_samples": Y_1_samples,
        }

    def estimate_ate(self, X, n_samples=100):
        results = self.estimate_cate(X, n_samples)
        ate = results["cate"].mean()

        if n_samples > 1:
            ate_se = results["cate"].std() / np.sqrt(len(results["cate"]))
            ci_lower = ate - 1.96 * ate_se
            ci_upper = ate + 1.96 * ate_se
            ci_95 = (ci_lower, ci_upper)
        else:
            ate_se = None
            ci_95 = None

        return {"ate": ate, "ate_se": ate_se, "ci_95": ci_95}


def save_estimator(estimator: WGANmodel, save_dir: str | Path) -> None:
    """Save model weights, optimizer states, scalers, and configuration."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "gen_control": estimator.wgan_control.generator.state_dict(),
            "crit_control": estimator.wgan_control.critic.state_dict(),
            "gen_treated": estimator.wgan_treated.generator.state_dict(),
            "crit_treated": estimator.wgan_treated.critic.state_dict(),
        },
        save_dir / "wgan_weights.pt",
    )

    torch.save(
        {
            "control": {
                "g_optimizer": estimator.wgan_control.g_optimizer.state_dict(),
                "c_optimizer": estimator.wgan_control.c_optimizer.state_dict(),
            },
            "treated": {
                "g_optimizer": estimator.wgan_treated.g_optimizer.state_dict(),
                "c_optimizer": estimator.wgan_treated.c_optimizer.state_dict(),
            },
        },
        save_dir / "optimizers.pt",
    )

    torch.save(
        {
            "covariate_scaler": estimator.covariate_scaler,
            "outcome_scaler_0": estimator.outcome_scaler_0,
            "outcome_scaler_1": estimator.outcome_scaler_1,
        },
        save_dir / "scalers.pt",
    )

    config = {
        "covariate_dim": estimator.wgan_control.generator.hidden[0].in_features - estimator.noise_dim,
        "noise_dim": estimator.noise_dim,
        "hidden_dim": estimator.wgan_control.generator.hidden[0].out_features,
        "n_layers": len(estimator.wgan_control.generator.hidden) // 3,
        "lambda_gp": estimator.wgan_control.lambda_gp,
        "n_critic": estimator.wgan_control.n_critic,
        "target_ate": estimator.target_ate,
        "lambda_ate": estimator.lambda_ate,
        "initialization_seed": estimator.initialization_seed,
        "training_seed": estimator.training_seed,
    }

    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def export_estimator_payload(estimator: WGANmodel) -> dict:
    """Build a spawn-safe in-memory payload for a fitted estimator."""
    config = {
        "covariate_dim": estimator.wgan_control.generator.hidden[0].in_features - estimator.noise_dim,
        "noise_dim": estimator.noise_dim,
        "hidden_dim": estimator.wgan_control.generator.hidden[0].out_features,
        "n_layers": len(estimator.wgan_control.generator.hidden) // 3,
        "lambda_gp": estimator.wgan_control.lambda_gp,
        "n_critic": estimator.wgan_control.n_critic,
        "target_ate": estimator.target_ate,
        "lambda_ate": estimator.lambda_ate,
        "initialization_seed": estimator.initialization_seed,
        "training_seed": estimator.training_seed,
    }

    def _state_to_numpy(state_dict):
        return {k: v.detach().cpu().numpy().copy() for k, v in state_dict.items()}

    return {
        "config": config,
        "weights": {
            "gen_control": _state_to_numpy(estimator.wgan_control.generator.state_dict()),
            "crit_control": _state_to_numpy(estimator.wgan_control.critic.state_dict()),
            "gen_treated": _state_to_numpy(estimator.wgan_treated.generator.state_dict()),
            "crit_treated": _state_to_numpy(estimator.wgan_treated.critic.state_dict()),
        },
        "scalers": {
            "covariate_scaler": estimator.covariate_scaler,
            "outcome_scaler_0": estimator.outcome_scaler_0,
            "outcome_scaler_1": estimator.outcome_scaler_1,
        },
    }


def load_estimator_payload(payload: dict, map_location: str | torch.device = "cpu") -> WGANmodel:
    """Reconstruct an estimator from an in-memory payload."""
    cfg = payload["config"]

    estimator = WGANmodel(
        covariate_dim=cfg["covariate_dim"],
        noise_dim=cfg["noise_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        lambda_gp=cfg["lambda_gp"],
        n_critic=cfg["n_critic"],
        target_ate=cfg.get("target_ate"),
        lambda_ate=cfg.get("lambda_ate", 1.0),
        initialization_seed=cfg.get("initialization_seed"),
    )
    estimator.training_seed = cfg.get("training_seed")

    def _state_from_numpy(state_dict_numpy):
        return {k: torch.tensor(v, device="cpu") for k, v in state_dict_numpy.items()}

    weights = payload["weights"]
    estimator.wgan_control.generator.load_state_dict(_state_from_numpy(weights["gen_control"]))
    estimator.wgan_control.critic.load_state_dict(_state_from_numpy(weights["crit_control"]))
    estimator.wgan_treated.generator.load_state_dict(_state_from_numpy(weights["gen_treated"]))
    estimator.wgan_treated.critic.load_state_dict(_state_from_numpy(weights["crit_treated"]))

    scalers = payload["scalers"]
    estimator.covariate_scaler = scalers["covariate_scaler"]
    estimator.outcome_scaler_0 = scalers["outcome_scaler_0"]
    estimator.outcome_scaler_1 = scalers["outcome_scaler_1"]

    estimator.wgan_control.set_device(str(map_location))
    estimator.wgan_treated.set_device(str(map_location))

    return estimator


def load_estimator(save_dir: str | Path, map_location: str | torch.device = "cpu") -> WGANmodel:
    """Load a saved estimator from a checkpoint directory."""
    save_dir = Path(save_dir)

    with open(save_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    estimator = WGANmodel(
        covariate_dim=cfg["covariate_dim"],
        noise_dim=cfg["noise_dim"],
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        lambda_gp=cfg["lambda_gp"],
        n_critic=cfg["n_critic"],
        target_ate=cfg.get("target_ate"),
        lambda_ate=cfg.get("lambda_ate", 1.0),
        initialization_seed=cfg.get("initialization_seed"),
    )
    estimator.training_seed = cfg.get("training_seed")

    weights = torch.load(save_dir / "wgan_weights.pt", map_location=map_location)
    estimator.wgan_control.generator.load_state_dict(weights["gen_control"])
    estimator.wgan_control.critic.load_state_dict(weights["crit_control"])
    estimator.wgan_treated.generator.load_state_dict(weights["gen_treated"])
    estimator.wgan_treated.critic.load_state_dict(weights["crit_treated"])

    optimizers = torch.load(save_dir / "optimizers.pt", map_location=map_location)
    estimator.wgan_control.g_optimizer.load_state_dict(optimizers["control"]["g_optimizer"])
    estimator.wgan_control.c_optimizer.load_state_dict(optimizers["control"]["c_optimizer"])
    estimator.wgan_treated.g_optimizer.load_state_dict(optimizers["treated"]["g_optimizer"])
    estimator.wgan_treated.c_optimizer.load_state_dict(optimizers["treated"]["c_optimizer"])

    scalers = torch.load(save_dir / "scalers.pt", map_location=map_location)
    estimator.covariate_scaler = scalers["covariate_scaler"]
    estimator.outcome_scaler_0 = scalers["outcome_scaler_0"]
    estimator.outcome_scaler_1 = scalers["outcome_scaler_1"]

    return estimator


def generate_observed_outcomes(
    estimator: WGANmodel,
    X: np.ndarray | torch.Tensor,
    A: np.ndarray | torch.Tensor,
    seed: int | None = None,
) -> np.ndarray:
    """Generate one synthetic observed outcome per row using observed treatment A."""
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    if torch.is_tensor(A):
        A_np = A.detach().cpu().numpy().reshape(-1)
    else:
        A_np = np.asarray(A).reshape(-1)

    y0 = estimator.sample_y0(X, n_samples=1, seed=seed)
    y1 = estimator.sample_y1(X, n_samples=1, seed=None if seed is None else int(seed) + 1)
    return np.where(A_np == 1, y1, y0).reshape(-1)


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

        rows.append(
            {
                "generator": str(label),
                "group": str(grp_name),
                "W1": float(wasserstein_distance(yr, yg)),
                "KS": float(ks_2samp(yr, yg).statistic),
                "Energy": float(energy_distance(yr, yg)),
                "JS_hist": float(js_divergence_hist(yr, yg, bins="fd")),
                "MeanDiff": float(np.mean(yg) - np.mean(yr)),
                "StdRatio": float(np.std(yg, ddof=1) / (np.std(yr, ddof=1) + 1e-12)),
            }
        )
    return rows


def _read_table(data_path: str | Path) -> pd.DataFrame:
    path = Path(data_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

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


def _choose_bivariates(X_df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        return [c for c in requested if c in X_df.columns][:3]
    return X_df.columns.tolist()[:3]


def run_wgan_standalone_diagnostics(
    *,
    estimator: WGANmodel,
    X_df: pd.DataFrame,
    A_np,
    Y_np,
    out_dir: str | Path,
    out_prefix: str = "wgan",
    generator_label: str = "WGAN",
    seed_sample: int = 12345,
    n_samples_mc: int = 1000,
    bivar_cols: list[str] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bivar_cols is None:
        bivar_cols = []

    X_np = X_df.to_numpy(dtype=np.float32)
    A_np = np.asarray(A_np).reshape(-1).astype(int)
    Y_np = np.asarray(Y_np, dtype=float).reshape(-1)

    y0_draws = estimator.sample_y0(X_np, n_samples=n_samples_mc, seed=seed_sample)
    y1_draws = estimator.sample_y1(X_np, n_samples=n_samples_mc, seed=seed_sample + 1)

    if y0_draws.ndim == 1:
        y0_draws = y0_draws[:, None]
    if y1_draws.ndim == 1:
        y1_draws = y1_draws[:, None]

    mu0_hat = y0_draws.mean(axis=1)
    mu1_hat = y1_draws.mean(axis=1)
    sigma0_hat = y0_draws.std(axis=1, ddof=1)
    sigma1_hat = y1_draws.std(axis=1, ddof=1)
    cate_hat = mu1_hat - mu0_hat
    y_obs_syn = np.where(A_np == 1, y1_draws[:, 0], y0_draws[:, 0])

    cate_results = estimator.estimate_cate(X_np, n_samples=min(100, n_samples_mc))
    ate_results = estimator.estimate_ate(X_np, n_samples=min(100, n_samples_mc))

    correlation_results = {
        "variable": [],
        "real_corr": [],
        "synthetic_corr": [],
        "difference": [],
        "abs_difference": [],
    }

    for col in X_df.columns.tolist():
        x_col = np.asarray(X_df[col].values).ravel()
        real_corr = float(np.corrcoef(x_col, Y_np)[0, 1])
        synthetic_corr = float(np.corrcoef(x_col, y_obs_syn)[0, 1])
        diff = synthetic_corr - real_corr

        correlation_results["variable"].append(col)
        correlation_results["real_corr"].append(real_corr)
        correlation_results["synthetic_corr"].append(synthetic_corr)
        correlation_results["difference"].append(diff)
        correlation_results["abs_difference"].append(abs(diff))

    real_corr_a = float(np.corrcoef(A_np, Y_np)[0, 1])
    synthetic_corr_a = float(np.corrcoef(A_np, y_obs_syn)[0, 1])
    diff_a = synthetic_corr_a - real_corr_a
    correlation_results["variable"].append("Treatment (A)")
    correlation_results["real_corr"].append(real_corr_a)
    correlation_results["synthetic_corr"].append(synthetic_corr_a)
    correlation_results["difference"].append(diff_a)
    correlation_results["abs_difference"].append(abs(diff_a))

    correlation_df = pd.DataFrame(correlation_results)
    mean_abs_diff = float(correlation_df["abs_difference"].mean())
    max_abs_diff = float(correlation_df["abs_difference"].max())

    dist_df = pd.DataFrame(
        distribution_distances(
            y_gen=y_obs_syn,
            y_ref=Y_np,
            A_np=A_np,
            label=generator_label,
        )
    )

    rowlevel_df = pd.DataFrame(
        {
            "A": A_np,
            "Y_real": Y_np,
            "Y_synthetic": y_obs_syn,
            "mu0_hat": mu0_hat,
            "mu1_hat": mu1_hat,
            "sigma0_hat": sigma0_hat,
            "sigma1_hat": sigma1_hat,
            "cate_hat": cate_hat,
            "y0_mean": np.asarray(cate_results["y0_mean"]).reshape(-1),
            "y1_mean": np.asarray(cate_results["y1_mean"]).reshape(-1),
        },
        index=X_df.index,
    )

    results_df = pd.DataFrame(
        {
            "cate": np.asarray(cate_results["cate"]).reshape(-1),
            "cate_se": (
                np.full(len(X_df), np.nan)
                if cate_results["cate_se"] is None
                else np.asarray(cate_results["cate_se"]).reshape(-1)
            ),
            "y0_predicted": np.asarray(cate_results["y0_mean"]).reshape(-1),
            "y1_predicted": np.asarray(cate_results["y1_mean"]).reshape(-1),
            "y_synthetic": y_obs_syn,
            "observed_treatment": A_np,
            "observed_outcome": Y_np,
        }
    )
    results_df = pd.concat([results_df, X_df.reset_index(drop=True)], axis=1)

    control_history = list(getattr(getattr(estimator, "wgan_control", None), "training_history", []) or [])
    treated_history = list(getattr(getattr(estimator, "wgan_treated", None), "training_history", []) or [])

    training_metrics = {
        "epoch": [],
        "control_wdist": [],
        "control_c_loss": [],
        "control_g_loss": [],
        "control_gp": [],
        "treated_wdist": [],
        "treated_c_loss": [],
        "treated_g_loss": [],
        "treated_gp": [],
    }
    n_epochs_hist = min(len(control_history), len(treated_history))
    for i in range(n_epochs_hist):
        training_metrics["epoch"].append(i + 1)
        training_metrics["control_wdist"].append(control_history[i].get("wasserstein_distance", np.nan))
        training_metrics["control_c_loss"].append(control_history[i].get("c_loss", np.nan))
        training_metrics["control_g_loss"].append(control_history[i].get("g_loss", np.nan))
        training_metrics["control_gp"].append(control_history[i].get("gp", np.nan))
        training_metrics["treated_wdist"].append(treated_history[i].get("wasserstein_distance", np.nan))
        training_metrics["treated_c_loss"].append(treated_history[i].get("c_loss", np.nan))
        training_metrics["treated_g_loss"].append(treated_history[i].get("g_loss", np.nan))
        training_metrics["treated_gp"].append(treated_history[i].get("gp", np.nan))
    training_df = pd.DataFrame(training_metrics)

    checkpoint_rows = []
    for arm_label, log in (
        ("control", list(getattr(estimator, "checkpoint_log_control", []) or [])),
        ("treated", list(getattr(estimator, "checkpoint_log_treated", []) or [])),
    ):
        for row in log:
            checkpoint_rows.append(
                {
                    "arm": arm_label,
                    "epoch": int(row.get("epoch")),
                    "mean_w1_scaled": float(row.get("mean_w1_scaled")),
                }
            )
    checkpoint_df = pd.DataFrame(checkpoint_rows)

    summary_stats = {
        "generator_label": generator_label,
        "n_obs": int(len(X_df)),
        "n_covariates": int(X_df.shape[1]),
        "n_control": int((A_np == 0).sum()),
        "n_treated": int((A_np == 1).sum()),
        "noise_dim": int(estimator.noise_dim),
        "lambda_gp": float(estimator.wgan_control.lambda_gp),
        "n_critic": int(estimator.wgan_control.n_critic),
        "best_epoch_control": estimator.best_epoch_control,
        "best_epoch_treated": estimator.best_epoch_treated,
        "best_w1_scaled_control": estimator.best_w1_scaled_control,
        "best_w1_scaled_treated": estimator.best_w1_scaled_treated,
        "ate": float(ate_results["ate"]),
        "ate_se": np.nan if ate_results["ate_se"] is None else float(ate_results["ate_se"]),
        "ate_ci_lower": (
            np.nan if ate_results["ci_95"] is None else float(ate_results["ci_95"][0])
        ),
        "ate_ci_upper": (
            np.nan if ate_results["ci_95"] is None else float(ate_results["ci_95"][1])
        ),
        "mean_cate": float(np.mean(np.asarray(cate_results["cate"]).reshape(-1))),
        "std_cate": float(np.std(np.asarray(cate_results["cate"]).reshape(-1))),
        "mean_real_outcome": float(np.mean(Y_np)),
        "std_real_outcome": float(np.std(Y_np, ddof=1)),
        "mean_synthetic_outcome": float(np.mean(y_obs_syn)),
        "std_synthetic_outcome": float(np.std(y_obs_syn, ddof=1)),
        "mean_abs_corr_diff": mean_abs_diff,
        "max_abs_corr_diff": max_abs_diff,
    }

    training_csv = out_dir / f"{out_prefix}_training_metrics.csv"
    corr_csv = out_dir / f"{out_prefix}_correlation_analysis.csv"
    dist_csv = out_dir / f"{out_prefix}_distribution_distances.csv"
    indiv_csv = out_dir / f"{out_prefix}_individual_results.csv"
    rowlevel_csv = out_dir / f"{out_prefix}_diagnostics_rowlevel.csv"
    summary_csv = out_dir / f"{out_prefix}_summary_statistics.csv"
    checkpoint_csv = out_dir / f"{out_prefix}_checkpoint_wdist.csv"
    summary_json = out_dir / f"{out_prefix}_summary.json"
    diagnostics_txt = out_dir / f"{out_prefix}_diagnostics_summary.txt"
    report_txt = out_dir / f"{out_prefix}_REPORT.txt"
    diag_png = out_dir / f"{out_prefix}_diagnostics.png"
    bivar_png = out_dir / f"{out_prefix}_bicovariate_diagnostics.png"
    model_dir = out_dir / "trained_wgan"

    training_df.to_csv(training_csv, index=False)
    correlation_df.to_csv(corr_csv, index=False)
    dist_df.to_csv(dist_csv, index=False)
    results_df.to_csv(indiv_csv, index=False)
    rowlevel_df.to_csv(rowlevel_csv, index=True)
    pd.DataFrame([summary_stats]).to_csv(summary_csv, index=False)
    if not checkpoint_df.empty:
        checkpoint_df.to_csv(checkpoint_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    with open(diagnostics_txt, "w", encoding="utf-8") as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    if not training_df.empty:
        ax1.plot(training_df["epoch"], training_df["control_wdist"], label="Control")
        ax1.plot(training_df["epoch"], training_df["treated_wdist"], label="Treated")
        ax1.legend()
    ax1.set_title("Training W-Distance")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("W-distance proxy")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if not training_df.empty:
        ax2.plot(training_df["epoch"], training_df["control_g_loss"], label="Control G")
        ax2.plot(training_df["epoch"], training_df["control_c_loss"], label="Control C")
        ax2.plot(training_df["epoch"], training_df["treated_g_loss"], label="Treated G", linestyle="--")
        ax2.plot(training_df["epoch"], training_df["treated_c_loss"], label="Treated C", linestyle="--")
        ax2.legend()
    ax2.set_title("Training Losses")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(Y_np, bins=40, alpha=0.6, density=True, label="Observed")
    ax3.hist(y_obs_syn, bins=40, alpha=0.6, density=True, label="Synthetic")
    ax3.set_title("Overall Outcome Distribution")
    ax3.set_xlabel("Outcome")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(Y_np[A_np == 0], bins=35, alpha=0.6, density=True, label="Observed")
    ax4.hist(y_obs_syn[A_np == 0], bins=35, alpha=0.6, density=True, label="Synthetic")
    ax4.set_title("Control Group Distribution")
    ax4.set_xlabel("Outcome")
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(Y_np[A_np == 1], bins=35, alpha=0.6, density=True, label="Observed")
    ax5.hist(y_obs_syn[A_np == 1], bins=35, alpha=0.6, density=True, label="Synthetic")
    ax5.set_title("Treated Group Distribution")
    ax5.set_xlabel("Outcome")
    ax5.set_ylabel("Density")
    ax5.legend()
    ax5.grid(alpha=0.3)

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

    if len(bivar_cols) >= 3 and all(col in X_df.columns for col in bivar_cols[:3]):
        x1_name, x2_name, x3_name = bivar_cols[:3]
        x1 = X_df[x1_name].to_numpy()
        x2 = X_df[x2_name].to_numpy()
        x3 = X_df[x3_name].to_numpy()

        vmin = float(np.nanmin([Y_np.min(), y_obs_syn.min()]))
        vmax = float(np.nanmax([Y_np.max(), y_obs_syn.max()]))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)
        fig.suptitle(f"Bi-Covariate Diagnostics: Real vs {generator_label}-Synthetic", fontsize=18, y=0.98)
        pairs = [(x1_name, x1, x2_name, x2), (x1_name, x1, x3_name, x3), (x2_name, x2, x3_name, x3)]

        def scatter_panel(ax, x, y, c, title):
            sc = ax.scatter(x, y, c=c, s=18, alpha=0.7, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=11)
            ax.grid(alpha=0.25)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("Outcome")

        for j, (xname, x, yname, y) in enumerate(pairs):
            scatter_panel(axes[0, j], x, y, Y_np, f"Real: {xname} vs {yname}")
        for j, (xname, x, yname, y) in enumerate(pairs):
            scatter_panel(axes[1, j], x, y, y_obs_syn, f"{generator_label}-Synthetic: {xname} vs {yname}")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(bivar_png, dpi=200, bbox_inches="tight")
        plt.close(fig)

    checkpoint_lines = []
    if checkpoint_df.empty:
        checkpoint_lines.append("(no checkpoint log available)")
    else:
        checkpoint_lines.append("CONTROL CHECKPOINTS")
        checkpoint_lines.append("-" * 80)
        checkpoint_lines.append(
            checkpoint_df.loc[checkpoint_df["arm"] == "control", ["epoch", "mean_w1_scaled"]].to_string(index=False)
        )
        checkpoint_lines.append("")
        checkpoint_lines.append("TREATED CHECKPOINTS")
        checkpoint_lines.append("-" * 80)
        checkpoint_lines.append(
            checkpoint_df.loc[checkpoint_df["arm"] == "treated", ["epoch", "mean_w1_scaled"]].to_string(index=False)
        )

    report = f"""
{'=' * 80}
WGAN TRAINING REPORT
{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SETTINGS
- out_prefix: {out_prefix}
- generator_label: {generator_label}
- n_obs: {len(X_df)}
- n_samples_mc: {n_samples_mc}
- seed_sample: {seed_sample}

BEST CHECKPOINTS
- Control best epoch: {estimator.best_epoch_control}
- Control best mean exact W1 (scaled): {estimator.best_w1_scaled_control}
- Treated best epoch: {estimator.best_epoch_treated}
- Treated best mean exact W1 (scaled): {estimator.best_w1_scaled_treated}

OBSERVED VS SYNTHETIC OUTCOMES
- Mean observed outcome:  {float(np.mean(Y_np)):.6f}
- Mean synthetic outcome: {float(np.mean(y_obs_syn)):.6f}
- SD observed outcome:    {float(np.std(Y_np, ddof=1)):.6f}
- SD synthetic outcome:   {float(np.std(y_obs_syn, ddof=1)):.6f}
- ATE:                    {float(ate_results['ate']):.6f}
- ATE SE:                 {np.nan if ate_results['ate_se'] is None else float(ate_results['ate_se']):.6f}
- Mean abs corr diff:     {mean_abs_diff:.6f}
- Max abs corr diff:      {max_abs_diff:.6f}

CHECKPOINT EVOLUTION OF W-DISTANCE
{'-' * 80}
{chr(10).join(checkpoint_lines)}

FILES GENERATED
- {training_csv}
- {corr_csv}
- {dist_csv}
- {indiv_csv}
- {rowlevel_csv}
- {summary_csv}
- {summary_json}
- {diagnostics_txt}
- {report_txt}
- {diag_png}
- {bivar_png if bivar_png.exists() else "(bicovariate figure skipped)"}
- {checkpoint_csv if checkpoint_csv.exists() else "(checkpoint csv skipped)"}
- {model_dir}
"""
    report_txt.write_text(report.strip() + "\n", encoding="utf-8")

    save_estimator(estimator, model_dir)

    return {
        "training_csv": training_csv,
        "corr_csv": corr_csv,
        "dist_csv": dist_csv,
        "indiv_csv": indiv_csv,
        "rowlevel_csv": rowlevel_csv,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "diagnostics_txt": diagnostics_txt,
        "report_txt": report_txt,
        "diag_png": diag_png,
        "bivar_png": bivar_png if bivar_png.exists() else None,
        "checkpoint_csv": checkpoint_csv if checkpoint_csv.exists() else None,
        "model_dir": model_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="variables_numeric")
    parser.add_argument("--outcome-col", type=str, required=True)
    parser.add_argument("--treatment-col", type=str, required=True)
    parser.add_argument("--drop-cols", nargs="*", default=[])
    parser.add_argument("--out-dir", type=str, default="output_wgan")
    parser.add_argument("--run-label", type=str, default="wgan")
    parser.add_argument("--bivar-cols", nargs="*", default=None)
    parser.add_argument("--n-epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--noise-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--checkpoint-n-draws", type=int, default=5)
    parser.add_argument("--n-samples-mc", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    df = _read_table(args.data_path)
    required_cols = {args.outcome_col, args.treatment_col}
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    drop_cols = [c for c in args.drop_cols if c in df.columns and c not in required_cols]
    X_df = df.drop(columns=[args.outcome_col, args.treatment_col, *drop_cols]).copy()
    A_np = np.rint(df[args.treatment_col].to_numpy(dtype=float).reshape(-1)).astype(int)
    if not set(np.unique(A_np)).issubset({0, 1}):
        raise ValueError("Treatment column must be binary and coded as {0,1}.")
    Y_np = df[args.outcome_col].to_numpy(dtype=float).reshape(-1)

    bivar_cols = _choose_bivariates(X_df, args.bivar_cols)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = WGANmodel(
        covariate_dim=X_df.shape[1],
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lambda_gp=10.0,
        n_critic=5,
        g_lr=5e-5,
        c_lr=2e-4,
        target_ate=None,
        lambda_ate=1.0,
        initialization_seed=42,
    )

    estimator.fit(
        X=X_df.to_numpy(dtype=np.float32),
        A=A_np.astype(np.float32),
        Y=Y_np.astype(np.float32),
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        print_every=max(1, min(500, args.checkpoint_every)),
        checkpoint_every=args.checkpoint_every,
        checkpoint_n_draws=args.checkpoint_n_draws,
    )

    outputs = run_wgan_standalone_diagnostics(
        estimator=estimator,
        X_df=X_df,
        A_np=A_np,
        Y_np=Y_np,
        out_dir=out_dir,
        out_prefix=args.run_label,
        generator_label="WGAN",
        seed_sample=12345,
        n_samples_mc=args.n_samples_mc,
        bivar_cols=bivar_cols,
    )

    print("\nANALYSIS COMPLETE")
    print(f"Report: {outputs['report_txt']}")
    print(f"Diagnostics plot: {outputs['diag_png']}")
    print(f"Model dir: {outputs['model_dir']}")


if __name__ == "__main__":
    main()
