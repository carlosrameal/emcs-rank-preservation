# Running experiments

Install the simulation dependencies in a separate environment:

```
python -m pip install -r requirements-simulation.txt
```

The full study requires stochtree (BCF) and PyTorch (NN and WGAN).
The complete pinned environment has not been installed and retested while
preparing this draft.

## Small demonstration

```
python run_experiments.py --demo
```

This generates one N&W A reference dataset with 100 observations, fits the OLS2
simulator and two estimators, and evaluates five outcome redraws under both
target and simulation benchmarks. It checks execution, not substantive results.
Output goes to `_build/experiments/demo/seed2001`.

## Manuscript settings

```
python run_experiments.py --setting nie-A-1000 --seed 2001
python run_experiments.py --setting nie-A-1000 --all-seeds
python run_experiments.py --setting acic-iid --acic-data PATH --all-seeds
```

N&W: choose A, B, C or D and n=1000 or 4302. ACIC: choose iid,
nonadditive, group_corr or heteroskedastic. The full settings use six N&W
covariates, noise scale 1, all five simulators, all fifteen estimators and 1000
Monte Carlo draws. Reference seeds are 2001 through 2100. WGAN defaults are
5000 epochs and batch size 512. `--workers` controls parallel execution.
`--dry-run` prints the command. Existing output folders are never overwritten.

ACIC requires the covariate table expected by `structured_dgps.py`.
Treatment is generated internally for each reference seed, after which
covariates and treatment are held fixed during benchmark evaluation. No
separate treatment-preparation script is needed. Input covariates are not
bundled; acquisition and preparation instructions remain to be completed.

These experiments generate fresh performance summaries. They do not overwrite
the archived final results. To reproduce the published tables, use
`python reproduce.py` instead. The full 1200-reference simulation study has not
been rerun while preparing the package.
