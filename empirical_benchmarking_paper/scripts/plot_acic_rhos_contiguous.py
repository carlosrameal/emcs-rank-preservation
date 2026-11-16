"""Render one four-panel ACIC rank-correlation figure."""

import os
from paper_results_config import RESULTS_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_DIR = str(RESULTS_ROOT / "2017")
OUT_DIR = os.path.join(BASE, "paper", "empirical_benchmarking_paper", "figures", "acic_rhos_contiguous")
OUTPUT = os.path.join(OUT_DIR, "acic_rhos_combined.png")

DGPS = (
    ("iid", "IID errors"),
    ("nonadditive", "Non-additive errors"),
    ("heteroskedastic", "Heteroskedastic errors"),
    ("group_corr", "Group-correlated errors"),
)
MODES = (("full_rho", "all"), ("excl_rho", "excl."))
SIMULATORS = ("OLS", "RF", "GB", "NN", "WGAN")
SIMULATOR_LABELS = {"OLS": "OLS2"}


def load(prefix: str) -> pd.DataFrame:
    folder = next(
        (
            name for name in sorted(os.listdir(RESULTS_DIR))
            if name.startswith(prefix + "__")
            and name.endswith("_analysis")
            and os.path.isfile(os.path.join(RESULTS_DIR, name, "per_seed_rho.csv"))
        ),
        None,
    )
    if folder is None:
        raise FileNotFoundError(f"No ACIC analysis found for {prefix}.")
    return pd.read_csv(os.path.join(RESULTS_DIR, folder, "per_seed_rho.csv"))


def layout():
    positions, labels, centers = [], [], []
    n_conditions = len(MODES)
    for simulator_index, _ in enumerate(SIMULATORS):
        for condition_index, (_, label) in enumerate(MODES, start=1):
            positions.append(simulator_index + condition_index / (n_conditions + 1))
            labels.append(label)
        centers.append(simulator_index + 0.5)
    return positions, labels, centers


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    positions, labels, centers = layout()
    fig, axes = plt.subplots(len(DGPS), 1, figsize=(17, 11.2), sharex=True)
    rng = np.random.default_rng(20260909)

    for row_index, (ax, (prefix, title)) in enumerate(zip(axes, DGPS)):
        frame = load(prefix)
        for simulator_index, simulator in enumerate(SIMULATORS):
            for mode_index, (column, _) in enumerate(MODES):
                position = positions[2 * simulator_index + mode_index]
                values = frame.loc[frame["generator"].eq(simulator), column].dropna().to_numpy(float)
                ax.scatter(
                    np.full(len(values), position) + rng.normal(0, 0.018, len(values)),
                    values,
                    s=10,
                    alpha=0.30,
                    linewidth=0,
                    color="#2F6DB5",
                )
                ax.hlines(np.mean(values), position - 0.075, position + 0.075, colors="black", linewidth=1.6)
        for split in range(1, len(SIMULATORS)):
            ax.axvline(split, color="0.85", linewidth=0.8)
        ax.axhline(0, color="0.65", linewidth=0.7)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(-1, 1)
        yticks = np.linspace(-1, 1, 5)
        ax.set_yticks(yticks)
        ytick_labels = [f"{tick:.1f}" for tick in yticks]
        ax.set_yticklabels(ytick_labels)
        ax.set_xlim(0, len(SIMULATORS))
        ax.tick_params(axis="y", labelsize=11)
        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.text(
            -0.10,
            0.5,
            title,
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(labels, fontsize=9.5)
    for center, simulator in zip(centers, SIMULATORS):
        axes[-1].text(
            center,
            -0.21,
            SIMULATOR_LABELS.get(simulator, simulator),
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            transform=axes[-1].get_xaxis_transform(),
        )
    fig.text(0.075, 0.5, "Spearman $\\rho$", rotation="vertical", va="center", fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.995, top=0.99, bottom=0.10, hspace=0.16)
    fig.savefig(OUTPUT, dpi=240)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
