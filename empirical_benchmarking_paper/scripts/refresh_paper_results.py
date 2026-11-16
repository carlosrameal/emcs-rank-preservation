"""Regenerate the empirical paper's analyses, tables and statistical figures."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import subprocess
import sys

from paper_results_config import PROJECT_ROOT, RESULTS_ROOT

SCRIPTS = Path(__file__).resolve().parent
LOGS = RESULTS_ROOT / "paper_refresh_logs"


def run(label, script, *args):
    LOGS.mkdir(exist_ok=True)
    with (LOGS / (label + ".log")).open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(script), *map(str, args)],
                                cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {LOGS / (label + '.log')}")
    print(f"Completed: {label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Rebuild paper assets from the existing validated analyzer CSVs.')
    args = parser.parse_args()
    bundles = sorted(RESULTS_ROOT.glob("2017/*.zip")) + sorted(RESULTS_ROOT.glob("Nie/*.zip"))
    assert len(bundles) == 12
    def analyze(bundle):
        run(bundle.stem, PROJECT_ROOT / "analyze_simulator_bundle.py", bundle,
            "--include-wgan-in-distance-corr", "--allow-mixed-estimands")
    if not args.skip_analysis:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(analyze, bundles))
    run("make_induced_estimand_table", SCRIPTS / "make_induced_estimand_table.py")
    scripts = ["make_gcomp_selection_table", "make_rank_aggregation_table",
               "plot_acic_rhos_contiguous", "plot_nie_rhos_contiguous", "plot_promotion_shift"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda name: run(name, SCRIPTS / (name + ".py")), scripts))
    run("compare_simulator_distances", SCRIPTS / "compare_simulator_distances.py")
    for excluded in ("WGAN", "RF"):
        run("distance_without_" + excluded.lower(), PROJECT_ROOT / "scripts/check_distance_sensitivity.py", "--exclude", excluded)
    run("make_distance_table", SCRIPTS / "make_distance_table.py")
    run("make_appendix_tables", SCRIPTS / "make_appendix_tables.py")
    # These checks are discussed in the manuscript; exploratory figures and
    # historical comparisons are intentionally outside the production rebuild.
    for name in ("check_kendall_sensitivity", "explore_mean_vs_median"):
        run(name, SCRIPTS / (name + ".py"))
    report = {"results_root": str(RESULTS_ROOT.relative_to(PROJECT_ROOT)),
              "analysis_configurations": len(bundles), "analysis_recomputed": not args.skip_analysis,
              "reference_datasets_per_setting": 100,
              "paper_generators": ["make_induced_estimand_table", *scripts, "make_distance_table", "make_appendix_tables",
                                   "compare_simulator_distances", "check_kendall_sensitivity", "explore_mean_vs_median"],
              "mixed_estimand_policy": "Preserve recorded per-estimator estimands and report existing conflicts."}
    (LOGS / "refresh_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
