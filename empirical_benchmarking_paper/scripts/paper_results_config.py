"""Single source of validated analysis inputs for the empirical paper."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "_build" / "results"

validation = json.loads((RESULTS_ROOT / "validation.json").read_text())
if (validation["complete_analysis_runs"] != 1200
        or validation["pending_runs"] != 0
        or not validation.get("analysis_seed_counts_match_manifest")):
    raise ValueError(f"Paper results require all 1200 validated runs: {RESULTS_ROOT}")


SETTINGS = (
    ("ACIC 2017", "IID", "--", RESULTS_ROOT / "2017" / "iid__mag1__noise1__conf1__tredraw__0_analysis"),
    ("ACIC 2017", "Non-additive", "--", RESULTS_ROOT / "2017" / "nonadditive__mag1__noise1__conf1__tredraw__0_analysis"),
    ("ACIC 2017", "Group-correlated", "--", RESULTS_ROOT / "2017" / "group_corr__mag1__noise1__conf1__tredraw__0_analysis"),
    ("ACIC 2017", "Heteroskedastic", "--", RESULTS_ROOT / "2017" / "heteroskedastic__mag1__noise1__conf1__tredraw__0_analysis"),
    ("Nie and Wager", "A", "1000", RESULTS_ROOT / "Nie" / "niea__n1000__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "A", "4302", RESULTS_ROOT / "Nie" / "niea__n4302__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "B", "1000", RESULTS_ROOT / "Nie" / "nieb__n1000__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "B", "4302", RESULTS_ROOT / "Nie" / "nieb__n4302__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "C", "1000", RESULTS_ROOT / "Nie" / "niec__n1000__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "C", "4302", RESULTS_ROOT / "Nie" / "niec__n4302__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "D", "1000", RESULTS_ROOT / "Nie" / "nied__n1000__d6__sigma1.0__0_analysis"),
    ("Nie and Wager", "D", "4302", RESULTS_ROOT / "Nie" / "nied__n4302__d6__sigma1.0__0_analysis"),
)

