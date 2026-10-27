"""Analysis of the POST HOC alignment check (separate Holm family F5: test MSE of the
eight feasible methods vs OLS over the 16 post hoc configurations)."""
import pickle, numpy as np, pandas as pd
import analyze as A
from feasible_target import feasible_targets
from run_posthoc_alignment import make_sigma_arranged

import os
if os.path.exists('results_posthoc_alignment.pkl'):
    res = pickle.load(open('results_posthoc_alignment.pkl', 'rb'))
    rows = []; cfg = []
    for c, order, rr in res:
        cfg.append(dict(c, order=order)); rows += rr
    df = pd.DataFrame(rows); cfg = pd.DataFrame(cfg)
    df.to_csv('replication_level_posthoc_alignment.csv.gz', index=False)
else:
    # replay from the archived replication-level records (same configuration list as the run script)
    df = pd.read_csv('replication_level_posthoc_alignment.csv.gz', float_precision='round_trip')
    cfg = []; cid = 2000
    for order in ('reversed', 'permuted'):
        for T in (20, 40, 100, 400):
            for rho in (0.3, 0.8):
                cfg.append(dict(cid=cid, p=30, T=T, rho=rho, v=20, mis=1, order=order)); cid += 1
    cfg = pd.DataFrame(cfg)
r = []; [r.extend(A.paired(df, 'test_mse', m, 'OLS')) for m in A.FEAS]
F = pd.DataFrame(r).rename(columns={'p': 'p_value'}); F['p_holm'] = A.holm(F.p_value.values)
F = F.merge(cfg[['cid', 'p', 'T', 'rho', 'v', 'mis', 'order']], on='cid')
F.to_csv('F5_posthoc_alignment_testMSE_vs_OLS.csv', index=False)
F['better'] = (F.p_holm < .05) & (F.mean_diff < 0); F['worse'] = (F.p_holm < .05) & (F.mean_diff > 0)
summ = F.groupby(['order', 'method']).agg(n=('cid', 'size'), better=('better', 'sum'), worse=('worse', 'sum'),
                                         med_rel=('rel_pct', 'median'), max_rel=('rel_pct', 'max')).reset_index()
o = df.groupby(['cid', 'method']).test_mse.mean().unstack()
orc = (100 * (o['Oracle'] / o['OLS'] - 1)).rename('oracle_dmse_pct').reset_index().merge(cfg, on='cid')
tg = []
for order in ('reversed', 'permuted'):
    for rho in (0.3, 0.8):
        S = make_sigma_arranged(30, rho, 20, order)
        t = feasible_targets(30, rho, 20, 0.25, Sigma=S)
        tg.append(dict(order=order, rho=rho, b0_slope=t['b0'][1], bS_slope=t['bS'][1], bF_slope=t['bF'][1],
                       excess_S=t['ex_S'], excess_F=t['ex_F']))
summ.to_csv('posthoc_alignment_summary.csv', index=False)
orc.to_csv('posthoc_alignment_oracle.csv', index=False)
pd.DataFrame(tg).to_csv('posthoc_alignment_targets.csv', index=False)
print(summ.round(2).to_string()); print(orc[['order', 'T', 'rho', 'oracle_dmse_pct']].round(2).to_string())
print(pd.DataFrame(tg).round(4).to_string())
