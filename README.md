# Empirical Monte Carlo benchmarking: publication draft

Code and final numerical results accompanying *Formalizing and Evaluating Rank
Preservation in Empirical Monte Carlo Studies of Causal Estimators*.

Authors: Carlos Ameal, Tom Loeys and Jan De Neve.

## Rebuild the paper's numerical results

Use Python 3.11 or later and install the analysis dependencies:

```
python -m pip install -r requirements-analysis.txt
python reproduce.py
```

The sole numerical input is `data/final_results.zip`. It contains complete
per-reference results, not just rounded paper tables. See `data/README.md` for
a description of the files and variables.
The command reconstructs analysis inputs under `_build/`, recomputes rankings,
regret and distance sensitivity checks, and regenerates the manuscript assets.
It does not fit simulation models. No original run archives are required.

## Compile the manuscript

After regeneration, change to `paper/empirical_benchmarking_paper` and run:

```
pdflatex Master.tex
bibtex Master
pdflatex Master.tex
pdflatex Master.tex
```

An installed LaTeX distribution with the packages requested by Master.tex is
required. The supplied Wiley template is used by the manuscript.

## Simulation code

The root Python modules implement data generation, simulator fitting and
estimator evaluation. `mc_study_main.py` is the experiment driver;
it also computes the target benchmark internally. Install
`requirements-simulation.txt` for experiments, or `requirements-analysis.txt`
for rebuilding tables and figures. See `SIMULATIONS.md` for commands.

This is a LOCAL DRAFT, not a published or fully documented simulation release.
Before publication: document acquisition/preparation of the ACIC covariates;
check third-party licensing and template redistribution; select a code licence;
and verify the full experiment environment. Re-running model fitting need not
reproduce floating-point results bit for bit across platforms.
