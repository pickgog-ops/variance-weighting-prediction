"""Post-run analysis additions (written after the verification of the main results).
Regenerates the family CSVs with unambiguous column names, and writes the
sensitivity family, target-shift summaries (including the feasible-FGLS limit),
condition numbers, covariance/precision error summaries with 95% CIs, oracle-gain
shares (unfiltered = protocol definition; filtered = post hoc, labelled), the
'best feasible method' counts and the post hoc alignment check."""
import pickle, numpy as np, pandas as pd
from scipy import stats
import analyze as A
from design import make_sigma, targets
from feasible_target import feasible_targets

def families_clean(df, cfg):
    F = {}
    r = []; [r.extend(A.paired(df, 'test_mse', m, 'OLS')) for m in A.FEAS]; F['F1_testMSE_vs_OLS'] = pd.DataFrame(r)
    r = []; [r.extend(A.paired(df, 'se_b0', m, 'OLS')) for m in A.FEAS]; F['F2_coefSE_vs_OLS'] = pd.DataFrame(r)
    r = []
    for m in A.ORIG:
        for ref in ['CV-power', 'CV-linear']:
            for met in ['test_mse', 'se_b0']:
                r.extend(A.paired(df, met, m, ref))
    F['F3_orig_vs_CV'] = pd.DataFrame(r)
    for k, v in F.items():
        v = v.rename(columns={'p': 'p_value'})
        v['p_holm'] = A.holm(v.p_value.values)
        F[k] = v.merge(cfg[['cid', 'p', 'T', 'rho', 'v', 'mis']], on='cid')
    return F

def ci_mean(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan, np.nan, np.nan
    m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h

if __name__ == '__main__':
    df, cfg = A.load('main')
    F = families_clean(df, cfg)
    for k, v in F.items():
        v.to_csv(k + '.csv', index=False)
    # sensitivity family F4
    ds, cs = A.load('sens')
    if A.has_pickle('sens'): ds.to_csv('replication_level_sens.csv.gz', index=False)
    r = []; [r.extend(A.paired(ds, 'test_mse', m, 'OLS')) for m in A.FEAS]
    F4 = pd.DataFrame(r).rename(columns={'p': 'p_value'}); F4['p_holm'] = A.holm(F4.p_value.values)
    F4 = F4.merge(cs[['cid', 'p', 'T', 'rho', 'v', 'mis']], on='cid'); F4.to_csv('F4_sens_a05_testMSE_vs_OLS.csv', index=False)
    # condition numbers
    cfg['cond'] = [np.linalg.cond(make_sigma(int(c.p), c.rho, c.v)) for _, c in cfg.iterrows()]
    cfg.to_csv('configs_with_condition_numbers.csv', index=False)
    # covariance / precision errors with CIs (both mean specifications)
    rows = []
    for (cid, m), g in df.groupby(['cid', 'method']):
        if m in ('OLS', 'Oracle'): continue
        cm, cl, ch = ci_mean(g.cov_err); pm, pl, ph = ci_mean(g.prec_err)
        rows.append(dict(cid=cid, method=m, cov_err=cm, cov_lo=cl, cov_hi=ch, prec_err_mean=pm, prec_lo=pl, prec_hi=ph,
                         prec_err_median=np.nanmedian(g.prec_err) if g.prec_err.notna().any() else np.nan,
                         n_prec=int(g.prec_err.notna().sum())))
    pd.DataFrame(rows).merge(cfg, on='cid').to_csv('cov_prec_errors_with_ci.csv', index=False)
    # target shift incl. feasible limit
    S = pd.read_csv('summary_main.csv')
    rows = []
    for _, c in cfg[cfg.mis == 1].iterrows():
        t = feasible_targets(int(c.p), c.rho, c.v, 0.25)
        sl = df[df.cid == c.cid].groupby('method').b_slope.mean()
        o = S[S.cid == c.cid].set_index('method').test_mse
        rows.append(dict(p=c.p, T=c['T'], rho=c.rho, v=c.v, b0_slope=t['b0'][1], bS_slope=t['bS'][1], bF_slope=t['bF'][1],
                         excess_S=t['ex_S'], excess_F=t['ex_F'], oracle_dmse_pct=100 * (o['Oracle'] / o['OLS'] - 1),
                         **{m: sl[m] for m in ['OLS', 'Oracle', 'Sample', 'Diag', 'LW-linear', 'OAS', 'LW-NL', 'QIS', 'CV-power', 'CV-linear'] if m in sl}))
    pd.DataFrame(rows).to_csv('target_shift_misspecified.csv', index=False)
    # oracle-gain shares
    c = S[S.mis == 0].pivot_table(index='cid', columns='method', values='se_b0')
    den = c['OLS'] - c['Oracle']; ok = den > 0.1 * c['OLS']
    sh = []
    for m in A.FEAS:
        s = 100 * (c['OLS'] - c[m]) / den
        sh.append(dict(method=m, median_all=np.nanmedian(s), n_all=int(s.notna().sum()),
                       median_filtered_posthoc=np.nanmedian(s[ok]), n_filtered=int(s[ok].notna().sum())))
    pd.DataFrame(sh).to_csv('oracle_gain_shares.csv', index=False)
    # best feasible method by coefficient MSE (correct mean)
    best = c[A.FEAS].idxmin(axis=1)
    bc = pd.DataFrame({'best': best}).join(cfg.set_index('cid')[['v']])
    bc.groupby(['v', 'best']).size().rename('n').reset_index().to_csv('best_feasible_counts.csv', index=False)
    # absolute coefficient RMSE vs beta0 and beta_Sigma (misspecified)
    S2 = S[S.mis == 1].copy(); S2['rmse_b0'] = np.sqrt(S2.se_b0); S2['rmse_bS'] = np.sqrt(S2.se_bS)
    S2[['cid', 'p', 'T', 'rho', 'v', 'method', 'rmse_b0', 'rmse_bS']].to_csv('coef_rmse_misspecified_both_targets.csv', index=False)
    print('ok')
