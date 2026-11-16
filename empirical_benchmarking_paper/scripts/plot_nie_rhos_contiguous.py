"""Render one four-panel Nie and Wager rank-correlation figure."""

import os
from paper_results_config import RESULTS_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS_DIR = str(RESULTS_ROOT / "Nie")
OUT_DIR = os.path.join(BASE, "paper", "empirical_benchmarking_paper", "figures", "nie_rhos_contiguous")
OUTPUT = os.path.join(OUT_DIR, "nie_rhos_combined.png")

DGPS = (("niea", "DGP A"), ("nieb", "DGP B"), ("niec", "DGP C"), ("nied", "DGP D"))
SAMPLES = (("n1000", "1k"), ("n4302", "4302"))
MODES = (("full_rho", "all"), ("excl_rho", "excl."))
SIMULATORS = ("OLS", "RF", "GB", "NN", "WGAN")
SIMULATOR_LABELS = {"OLS": "OLS2"}
SAMPLE_COLORS = {"n1000": "#2F6DB5", "n4302": "#D98C24"}


def load(prefix: str, sample: str) -> pd.DataFrame:
    folder = f"{prefix}__{sample}__d6__sigma1.0__0_analysis"
    path = os.path.join(RESULTS_DIR, folder, "per_seed_rho.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def layout():
    positions, labels, centers = [], [], []
    n_conditions = len(SAMPLES) * len(MODES)
    for simulator_index, _ in enumerate(SIMULATORS):
        condition_index = 1
        for _, sample_label in SAMPLES:
            for _, mode_label in MODES:
                positions.append(simulator_index + condition_index / (n_conditions + 1))
                labels.append(f"{sample_label}\n{mode_label}")
                condition_index += 1
        centers.append(simulator_index + 0.5)
    return positions, labels, centers


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    positions, labels, centers = layout()
    fig, axes = plt.subplots(4, 1, figsize=(17, 11.2), sharex=True)
    rng = np.random.default_rng(20260909)

    for row_index, (ax, (prefix, title)) in enumerate(zip(axes, DGPS)):
        frames = {sample: load(prefix, sample) for sample, _ in SAMPLES}
        for simulator_index, simulator in enumerate(SIMULATORS):
            for sample_index, (sample, _) in enumerate(SAMPLES):
                for mode_index, (column, _) in enumerate(MODES):
                    position = positions[4 * simulator_index + 2 * sample_index + mode_index]
                    values = frames[sample].loc[frames[sample]["generator"].eq(simulator), column].dropna().to_numpy(float)
                    ax.scatter(
                    np.full(len(values), position) + rng.normal(0, 0.018, len(values)),
                        values,
                        s=10,
                        alpha=0.30,
                        linewidth=0,
                        color=SAMPLE_COLORS[sample],
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
