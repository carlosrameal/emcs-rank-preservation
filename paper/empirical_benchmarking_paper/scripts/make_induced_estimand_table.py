from pathlib import Path
from paper_results_config import RESULTS_ROOT
import zipfile
import io
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'paper' / 'empirical_benchmarking_paper' / 'tables'
CONFIGS = [
    ('ACIC iid', RESULTS_ROOT / '2017/iid__mag1__noise1__conf1__tredraw.zip'),
    ('ACIC nonadditive', RESULTS_ROOT / '2017/nonadditive__mag1__noise1__conf1__tredraw.zip'),
    ('ACIC group-correlated', RESULTS_ROOT / '2017/group_corr__mag1__noise1__conf1__tredraw.zip'),
    ('ACIC heteroskedastic', RESULTS_ROOT / '2017/heteroskedastic__mag1__noise1__conf1__tredraw.zip'),
    (r'N\&W A ($n=1{,}000$)', RESULTS_ROOT / 'Nie/niea__n1000__d6__sigma1.0.zip'),
    (r'N\&W A ($n=4{,}302$)', RESULTS_ROOT / 'Nie/niea__n4302__d6__sigma1.0.zip'),
    (r'N\&W B ($n=1{,}000$)', RESULTS_ROOT / 'Nie/nieb__n1000__d6__sigma1.0.zip'),
    (r'N\&W B ($n=4{,}302$)', RESULTS_ROOT / 'Nie/nieb__n4302__d6__sigma1.0.zip'),
    (r'N\&W C ($n=1{,}000$)', RESULTS_ROOT / 'Nie/niec__n1000__d6__sigma1.0.zip'),
    (r'N\&W C ($n=4{,}302$)', RESULTS_ROOT / 'Nie/niec__n4302__d6__sigma1.0.zip'),
    (r'N\&W D ($n=1{,}000$)', RESULTS_ROOT / 'Nie/nied__n1000__d6__sigma1.0.zip'),
    (r'N\&W D ($n=4{,}302$)', RESULTS_ROOT / 'Nie/nied__n4302__d6__sigma1.0.zip'),
]
GENERATORS = [('REAL_DGP', 'Reference'), ('OLS', 'OLS2'), ('GB', 'GB'), ('RF', 'RF'), ('NN', 'NN'), ('WGAN', 'WGAN')]

def summaries(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('/MASTER_mc_summary.csv') and '/_dgp_benchmark/' not in n]
        bench_names = [n for n in z.namelist() if '/_dgp_benchmark/' in n and n.endswith('/MASTER_mc_summary.csv')]
        for name in names:
            run = name.split('/')[0]
            df = pd.read_csv(io.BytesIO(z.read(name)))
            bench = next((b for b in bench_names if b.split('/')[0] == run), None)
            if bench:
                ref = pd.read_csv(io.BytesIO(z.read(bench)))
                yield run, df, ref

def one_value(df, generator, *, average_conflicts=False):
    """Return the recorded estimand and its within-run distinct-value count."""
    vals = df.loc[df.generator.astype(str).eq(generator), 'ate_true'].dropna()
    counts = vals.value_counts()
    if counts.empty:
        raise ValueError(f'{generator}: no ate_true recorded')
    value = vals.mean() if average_conflicts else counts.index[0]
    return float(value), int(len(counts))

rows=[]
inconsistencies=[]
for label, path in CONFIGS:
    records=[]
    for run, df, ref in summaries(path):
        rec={'setting': label, 'seed': run.rsplit('seed', 1)[-1]}
        rec['Reference'], nvals = one_value(ref, 'REAL_DGP')
        if nvals != 1:
            inconsistencies.append((label, run, 'Reference', nvals))
        for generator, display in GENERATORS[1:]:
            rec[display], nvals = one_value(df, generator)
            if display == 'WGAN':
                values = df.loc[df.generator.eq('WGAN'), 'ate_true'].to_numpy()
                if len(values) != 15 or not np.allclose(values, values[0], rtol=1e-12, atol=1e-12):
                    raise ValueError(f'{label}, {run}: inconsistent new WGAN estimand')
            rec[display + '_gap'] = rec[display] - rec['Reference']
            rec[display + '_abs_gap'] = abs(rec[display + '_gap'])
            if nvals != 1:
                inconsistencies.append((label, run, display, nvals))
        records.append(rec)
    if len(records) != 100:
        raise ValueError(f'{label}: found {len(records)} complete runs')
    rows.extend(records)

raw=pd.DataFrame(rows)
raw.to_csv(OUT / 'induced_estimands_by_seed.csv', index=False)
means=raw.groupby('setting', sort=False)[[d for _,d in GENERATORS]].mean()
sds=raw.groupby('setting', sort=False)[[d for _,d in GENERATORS]].std(ddof=1)
summary=means.copy()
for c in means:
    summary[c+'_sd']=sds[c]
summary.to_csv(OUT / 'induced_estimands_summary.csv')
gap_columns = [display + suffix for _, display in GENERATORS[1:] for suffix in ('_gap', '_abs_gap')]
raw.groupby('setting', sort=False)[gap_columns].mean().to_csv(OUT / 'induced_estimand_gaps_summary.csv')
pd.DataFrame(inconsistencies, columns=['setting', 'run', 'generator', 'n_distinct_estimands']).to_csv(OUT / 'induced_estimands_integrity_issues.csv', index=False)

lines=[
 r'\begin{table*}[!t]',
 r'\centering',
 r'\caption{Reference and simulator-induced SACTEs, averaged over 100 reference datasets in each setting. Parentheses report across-dataset standard deviations. Each simulator is evaluated against its own induced SACTE. For each fitted WGAN and reference dataset, its induced SACTE is estimated by averaging 10,000 draws of each potential outcome over the observed covariates.}',
 r'\label{tab:induced-estimands}',
 r'\begin{tabular}{lrrrrrr}',
 r'\toprule',
 r'Setting & Reference & OLS2 & GB & RF & NN & WGAN \\',
 r'\midrule',
]
for label in means.index:
    vals=[]
    for _, col in GENERATORS:
        vals.append(f'{means.loc[label,col]:.3f} ({sds.loc[label,col]:.3f})')
    lines.append(label + ' & ' + ' & '.join(vals) + r' \\')
    if label == 'ACIC heteroskedastic':
        lines.append(r'\midrule')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
(OUT / 'induced_estimands.tex').write_text('\n'.join(lines), encoding='utf-8')
print(summary.round(4).to_string())
print('Wrote', OUT / 'induced_estimands.tex')
