# Final numerical results

Extract `final_results.zip` to inspect the CSV tables directly, or run
`python reproduce.py` from the package root to regenerate the manuscript results.

- performance.csv: 108,000 rows, one per setting, reference seed, benchmark and
  estimator; includes all five simulators and the target benchmark REAL_DGP.
- reference_values.csv: reference estimand values by setting, seed and benchmark.
- outcome_distances.csv: outcome-distance summaries by setting, seed, simulator
  and treatment group.
- settings.json: the twelve setting identifiers and their design families.

`evaluation_reference_value` is the reference used to calculate bias and MSE.
`mc_mean_ate`, `mc_bias`, `mc_var`, `mc_sd` and `mc_mse` are within-benchmark
Monte Carlo summaries; `N_MC` is the number of draws. Variance uses denominator
N_MC-1, so MSE = bias^2 + (N_MC-1)/N_MC * variance. Missing standard-error or
coverage quantities indicate measures not evaluated in this study. Simulator
identifier OLS denotes the OLS2 simulator; estimator names retain their code IDs.
Seeds 2001 through 2100 index the 100 reference datasets in each setting.
