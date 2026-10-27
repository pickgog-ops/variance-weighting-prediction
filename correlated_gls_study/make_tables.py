import pandas as pd, numpy as np
from design import make_sigma, targets
S=pd.read_csv('summary_main.csv')
F1=pd.read_csv('F1_testMSE_vs_OLS.csv'); F2=pd.read_csv('F2_coefSE_vs_OLS.csv'); F3=pd.read_csv('F3_orig_vs_CV.csv')
M=['Oracle','Sample','Diag','LW-linear','OAS','LW-NL','QIS','CV-linear','CV-power']
HDR=['Oracle','Sample','Diag','LW-lin.','OAS','LW-NL','QIS','CV-lin.','CV-pow.']
def mark(F,cid,m):
    r=F[(F.cid==cid)&(F.method==m)]
    if len(r)==0: return ''
    r=r.iloc[0]
    if r.p_holm<.05: return '$^{\\dagger}$' if r.mean_diff<0 else '$^{\\ddagger}$'
    return ''
def fmt(x,d=1):
    if pd.isna(x): return '--'
    s=f'{x:+.{d}f}'
    return s.replace('-','$-$')
def rel_table(metric,F,mis,absname,scale=1.0,d=1):
    lines=[]
    lines.append('\\begin{tabular}{@{}rrrrr r'+'r'*len(M)+'@{}}\n\\toprule')
    lines.append('$p$ & $T$ & $\\rho$ & $v$ & $\\kappa(\\Sigma)$ & '+absname+' & '+' & '.join(HDR)+' \\\\\n\\midrule')
    sub=S[S.mis==mis]
    prev=None
    for (p,T,rho,v),g in sub.groupby(['p','T','rho','v']):
        g=g.set_index('method'); cid=int(g.cid.iloc[0])
        base=g.loc['OLS',metric]
        cells=[]
        for m in M:
            val=g.loc[m,metric] if m in g.index else np.nan
            rel=100*(val/base-1) if not pd.isna(val) else np.nan
            cells.append(fmt(rel,d)+(mark(F,cid,m) if m!='Oracle' else ''))
        absv=np.sqrt(base) if metric.startswith('se') else base
        if prev is not None and prev!=(p,T): lines.append('\\addlinespace[1pt]')
        prev=(p,T)
        kap=np.linalg.cond(make_sigma(p,rho,v))
        lines.append(f'{p} & {T} & {rho:g} & {v:g} & {kap:.0f} & {absv*scale:.4f} & '+' & '.join(cells)+' \\\\')
    lines.append('\\bottomrule\n\\end{tabular}')
    return '\n'.join(lines)
open('tab_cgls_coef_correct.tex','w').write(rel_table('se_b0',F2,0,'RMSE$_{\\rm OLS}$'))
open('tab_cgls_coef_mis.tex','w').write(rel_table('se_b0',F2,1,'RMSE$_{\\rm OLS}$'))
open('tab_cgls_mse_correct.tex','w').write(rel_table('test_mse',F1,0,'MSE$_{\\rm OLS}$',d=2))
open('tab_cgls_mse_mis.tex','w').write(rel_table('test_mse',F1,1,'MSE$_{\\rm OLS}$',d=2))
# covariance / precision errors, correct mean, averaged over rho and v within (p,T)
E=['Sample','Diag','LW-linear','OAS','LW-NL','QIS','CV-linear','CV-power']
EH=['Sample','Diag','LW-lin.','OAS','LW-NL','QIS','CV-lin.','CV-pow.']
L=['\\begin{tabular}{@{}rr'+'r'*len(E)+'@{}}\n\\toprule',' & & \\multicolumn{%d}{c}{relative Frobenius error of $\\hat\\Sigma$ (correct mean)}\\\\'%len(E),
   '$p$ & $T$ & '+' & '.join(EH)+' \\\\\n\\midrule']
for metric,title in [('cov_err',None),('prec_err','relative Frobenius error of $\\hat\\Sigma^{-1}$ (correct mean)')]:
    if title: L.append('\\midrule\n & & \\multicolumn{%d}{c}{%s}\\\\\n\\midrule'%(len(E),title))
    sub=S[S.mis==0]
    for (p,T),g in sub.groupby(['p','T']):
        vals=[]
        for m in E:
            x=g[g.method==m][metric]
            vals.append('--' if x.isna().all() else f'{x.mean():.3f}')
        L.append(f'{p} & {T} & '+' & '.join(vals)+' \\\\')
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_coverr.tex','w').write('\n'.join(L))
# summary counts
def cnt(F,mis,m,ref=None,metric=None):
    f=F[(F.mis==mis)&(F.method==m)]
    if ref is not None: f=f[(f.ref==ref)&(f.metric==metric)]
    b=int(((f.p_holm<.05)&(f.mean_diff<0)).sum()); w=int(((f.p_holm<.05)&(f.mean_diff>0)).sum())
    return f'{b}/{w}', len(f)
L=['\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule',
   ' & \\multicolumn{2}{c}{test MSE vs OLS} & \\multicolumn{2}{c}{coef.\\ error vs OLS} & median share of & median \\\\',
   'Method & correct & missp. & correct & missp. & oracle gain (\\%) & time (ms) \\\\\n\\midrule']
c=S[S.mis==0].pivot_table(index='cid',columns='method',values='se_b0')
den=c['OLS']-c['Oracle']; ok=den>0.1*c['OLS']
for m in E:
    a,na=cnt(F1,0,m); b,nb=cnt(F1,1,m); cc,_=cnt(F2,0,m); dd,_=cnt(F2,1,m)
    sh=np.nanmedian((100*(c['OLS']-c[m])/den)[ok]) if m in c else np.nan
    t=S[S.method==m].time.median()*1000
    L.append(f'{m} ({na}) & {a} & {b} & {cc} & {dd} & {sh:.0f} & {t:.1f} \\\\')
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_summary.tex','w').write('\n'.join(L))
# analytic vs CV counts
L=['\\begin{tabular}{@{}llrrrr@{}}\n\\toprule',' & & \\multicolumn{2}{c}{test MSE} & \\multicolumn{2}{c}{coef.\\ error} \\\\',
   'Method & reference & correct & missp. & correct & missp. \\\\\n\\midrule']
for m in ['LW-linear','OAS','LW-NL','QIS']:
    for ref in ['CV-power','CV-linear']:
        cells=[cnt(F3,mis,m,ref,met)[0] for met in ['test_mse','se_b0'] for mis in (0,1)]
        cells=[cells[0],cells[1],cells[2],cells[3]]
        L.append(f'{m} & {ref} & '+' & '.join(cells)+' \\\\')
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_vscv.tex','w').write('\n'.join(L))
print('ok')

# ---------------- post-run additions ----------------
def abs_table(col, F=None):
    """Absolute coefficient RMSE under the misspecified mean, against beta0 (col='se_b0', with
    Holm marks of family 2 vs OLS) or against beta_Sigma (col='se_bS', descriptive)."""
    MM=['OLS']+M
    lines=['\\begin{tabular}{@{}rrrr'+'r'*len(MM)+'@{}}\n\\toprule',
           '$p$ & $T$ & $\\rho$ & $v$ & OLS & '+' & '.join(HDR)+' \\\\\n\\midrule']
    sub=S[S.mis==1]; prev=None
    for (p,T,rho,v),g in sub.groupby(['p','T','rho','v']):
        g=g.set_index('method'); cid=int(g.cid.iloc[0]); cells=[]
        for m in MM:
            val=np.sqrt(g.loc[m,col]) if m in g.index and not pd.isna(g.loc[m,col]) else np.nan
            mk=mark(F,cid,m) if (F is not None and m not in ('OLS','Oracle')) else ''
            cells.append(('--' if pd.isna(val) else f'{val:.3f}')+mk)
        if prev is not None and prev!=(p,T): lines.append('\\addlinespace[1pt]')
        prev=(p,T)
        lines.append(f'{p} & {T} & {rho:g} & {v:g} & '+' & '.join(cells)+' \\\\')
    lines.append('\\bottomrule\n\\end{tabular}')
    return '\n'.join(lines)
open('tab_cgls_coef_mis.tex','w').write(abs_table('se_b0',F2))
open('tab_cgls_coef_mis_bS.tex','w').write(abs_table('se_bS'))

# summary table with protocol-defined (unfiltered) oracle-gain share and the post hoc filtered share
SH=pd.read_csv('oracle_gain_shares.csv').set_index('method')
L=['\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule',
   ' & \\multicolumn{2}{c}{test MSE vs OLS} & \\multicolumn{2}{c}{coef.\\ error vs OLS} & \\multicolumn{2}{c}{median share of oracle gain (\\%)} & median \\\\',
   'Method ($n$) & correct & missp. & correct & missp. & all & post hoc$^{\\ast}$ & time (ms) \\\\\n\\midrule']
for m in E:
    a,na=cnt(F1,0,m); b,_=cnt(F1,1,m); cc,_=cnt(F2,0,m); dd,_=cnt(F2,1,m)
    t=S[S.method==m].time.median()*1000
    L.append(f"{m} ({na}) & {a} & {b} & {cc} & {dd} & {SH.loc[m,'median_all']:.0f} & {SH.loc[m,'median_filtered_posthoc']:.0f} & {t:.1f} \\\\")
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_summary.tex','w').write('\n'.join(L))

# target-shift table with the feasible (residual-second-moment) limit
R=pd.read_csv('target_shift_misspecified.csv')
L=['\\begin{tabular}{@{}rrr rrr rr r rrr@{}}\n\\toprule',
   ' & & & \\multicolumn{3}{c}{limiting slope} & \\multicolumn{2}{c}{excess risk} & Oracle & \\multicolumn{3}{c}{mean slope, $T=400$}\\\\',
   '$p$ & $\\rho$ & $v$ & $\\beta_{0,1}$ & $\\beta_{\\Sigma,1}$ & $\\beta_{\\tilde\\Sigma,1}$ & $\\beta_{\\Sigma}$ & $\\beta_{\\tilde\\Sigma}$ & $\\Delta$MSE (\\%) & LW-lin. & LW-NL & CV-pow. \\\\\n\\midrule']
for (p,rho,v),g in R.groupby(['p','rho','v']):
    g4=g[g['T']==400].iloc[0]
    L.append(f"{int(p)} & {rho:g} & {v:g} & {g4.b0_slope:.3f} & {g4.bS_slope:.3f} & {g4.bF_slope:.3f} & {g4.excess_S:.4f} & {g4.excess_F:.4f} & {fmt(g4.oracle_dmse_pct,2)} & {g4['LW-linear']:.3f} & {g4['LW-NL']:.3f} & {g4['CV-power']:.3f} \\\\")
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_shift.tex','w').write('\n'.join(L))

# covariance (mean) and precision (median) errors, both mean specifications
C=pd.read_csv('cov_prec_errors_with_ci.csv')
L=['\\begin{tabular}{@{}rr'+'r'*len(E)+'@{}}\n\\toprule','$p$ & $T$ & '+' & '.join(EH)+' \\\\']
for mis in (0,1):
    for metric,title in [('cov_err','mean relative Frobenius error of $\\hat\\Sigma$'),('prec_err_median','median relative Frobenius error of $\\hat\\Sigma^{-1}$')]:
        L.append('\\midrule\n & & \\multicolumn{%d}{c}{%s, %s}\\\\\n\\midrule'%(len(E),title,'correct mean' if mis==0 else 'omitted curvature'))
        sub=C[C.mis==mis]
        for (p,T),g in sub.groupby(['p','T']):
            vals=[]
            for m in E:
                x=g[g.method==m][metric]
                vals.append('--' if x.isna().all() else f'{x.mean():.3f}')
            L.append(f'{p} & {T} & '+' & '.join(vals)+' \\\\')
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_coverr.tex','w').write('\n'.join(L))
print('additions ok')

# post hoc alignment check (family F5)
PS=pd.read_csv('posthoc_alignment_summary.csv'); PO=pd.read_csv('posthoc_alignment_oracle.csv'); PT=pd.read_csv('posthoc_alignment_targets.csv')
L=['\\begin{tabular}{@{}l rr rr@{}}\n\\toprule',
   ' & \\multicolumn{2}{c}{reversed (anti-aligned)} & \\multicolumn{2}{c}{permuted (unaligned)} \\\\',
   'Method & better/worse & median (\\%) & better/worse & median (\\%) \\\\\n\\midrule']
for m in E:
    cells=[]
    for order in ('reversed','permuted'):
        r=PS[(PS.order==order)&(PS.method==m)].iloc[0]
        cells+= [f"{int(r.better)}/{int(r.worse)}", fmt(r.med_rel,2)]
    L.append(f'{m} & '+' & '.join(cells)+' \\\\')
L.append('\\midrule')
for order,lab in (('reversed','reversed'),('permuted','permuted')):
    o=PO[PO.order==order].oracle_dmse_pct; t=PT[PT.order==order]
    L.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{lab}: oracle $\\Delta$MSE {abs(o.min()) if abs(o.min())<0.005 else o.min():.2f} to {o.max():.2f}\\%; $\\beta_{{\\Sigma,1}}$ {t.bS_slope.min():.3f}--{t.bS_slope.max():.3f}, $\\beta_{{\\tilde\\Sigma,1}}$ {t.bF_slope.min():.3f}--{t.bF_slope.max():.3f}}} \\\\")
L.append('\\bottomrule\n\\end{tabular}')
open('tab_cgls_posthoc.tex','w').write('\n'.join(L))
print('posthoc ok')
