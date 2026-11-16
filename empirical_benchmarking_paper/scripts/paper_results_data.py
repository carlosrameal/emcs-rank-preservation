"""Validated data readers shared by manuscript tables and figures."""
from pathlib import Path
import numpy as np
import pandas as pd
from paper_results_config import SETTINGS

SIMULATORS = ("GB", "NN", "OLS", "RF")
DISPLAY = {"OLS": "OLS2"}

FAMILIES = {
    "ACIC 2017": tuple(directory for design, _, _, directory in SETTINGS if design == "ACIC 2017"),
    **{rf"N\&W DGP {letter.upper()}": tuple(
        directory for design, setting, _, directory in SETTINGS
        if design == "Nie and Wager" and setting == letter.upper()
    ) for letter in "abcd"},
}

ESTIMATORS = [
    "BCF_GCOMP", "GB_GB_AIPW", "GB_GB_TMLE", "GB_GCOMP", "GB_IPW",
    "GB_NN_AIPW", "GB_NN_TMLE", "GB_OLS2_AIPW", "GB_OLS2_TMLE",
    "GB_RF_AIPW", "GB_RF_TMLE", "NN_GCOMP", "OLS1", "OLS2_GCOMP", "RF_GCOMP",
]


DISPLAY_NAMES = {
    "BCF_GCOMP": "BCF G-comp", "GB_GB_AIPW": "GB--GB AIPW",
    "GB_GB_TMLE": "GB--GB TMLE", "GB_GCOMP": "GB G-comp",
    "GB_IPW": "GB IPW", "GB_NN_AIPW": "GB--NN AIPW",
    "GB_NN_TMLE": "GB--NN TMLE", "GB_OLS2_AIPW": "GB--OLS2 AIPW",
    "GB_OLS2_TMLE": "GB--OLS2 TMLE", "GB_RF_AIPW": "GB--RF AIPW",
    "GB_RF_TMLE": "GB--RF TMLE", "NN_GCOMP": "NN G-comp",
    "OLS1": "OLS1", "OLS2_GCOMP": "OLS2 G-comp",
    "RF_GCOMP": "RF G-comp",
}


def _read_mse(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "average_mc_mse_by_estimator_and_simulator.csv"
    frame = pd.read_csv(path)
    if len(frame) != len(ESTIMATORS):
        raise ValueError(f"Expected {len(ESTIMATORS)} rows in {path}, found {len(frame)}")
    if "estimator" in frame.columns:
        names = frame.pop("estimator").astype(str)
        if set(names) != set(ESTIMATORS):
            raise ValueError(f"Unexpected estimator set in {path}: {sorted(set(names))}")
        frame.index = names
    else:
        frame.index = ESTIMATORS
    return frame


def _read_rho(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "average_rho_by_generator.csv"
    frame = pd.read_csv(path)
    required = {"generator", "avg_full_rho", "sd_full_rho", "avg_excl_rho", "sd_excl_rho"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame


def promotion_by_type(directories: tuple[Path, ...]) -> pd.DataFrame:
    rows = []
    for directory in directories:
        detail = pd.read_csv(directory / "promotion_detail.csv")
        assert set(detail.seed) == set(range(2001, 2101)), directory
        assert detail.groupby(["seed", "generator"]).size().eq(15).all(), directory
        np.testing.assert_allclose(detail.promotion, detail.real_rank - detail.sim_rank)
        aligned = detail.family_aligned.astype(str).str.lower().eq("true")
        for simulator in SIMULATORS:
            group = detail.loc[detail.generator.eq(simulator)]
            learner = DISPLAY.get(simulator, simulator)
            expected = {"gcomp": learner + "_GCOMP", "aipw": "GB_" + learner + "_AIPW",
                        "tmle": "GB_" + learner + "_TMLE"}
            row = {"simulator": simulator}
            for kind, estimator in expected.items():
                selected = group.loc[group.estimator.eq(estimator)]
                assert len(selected) == (0 if simulator == "WGAN" else 100)
                row["promotion_" + kind] = selected.promotion.mean()
                row["shift_" + kind] = selected.promotion.abs().mean()
            row["promotion_rest"] = group.loc[~aligned.loc[group.index], "promotion"].mean()
            row["shift_rest"] = group.loc[~aligned.loc[group.index], "promotion"].abs().mean()
            rows.append(row)
    return pd.DataFrame(rows).groupby("simulator", sort=False).mean().reindex(SIMULATORS)

