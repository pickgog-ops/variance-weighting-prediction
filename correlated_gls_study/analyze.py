import pickle, numpy as np, pandas as pd, sys
from scipy import stats
FEAS=['Sample','Diag','LW-linear','OAS','LW-NL','QIS','CV-linear','CV-power']
ORIG=['LW-linear','OAS','LW-NL','QIS']
import os
def has_pickle(which): return os.path.exists(f'results_{which}.pkl')
def load(which):
    if not has_pickle(which):
        # replay from the archived replication-level records
        import lib
        df=pd.read_csv(f'replication_level_{which}.csv.gz',float_precision='round_trip')
        if which=='main': cfg=[dict(c,a=None) for c in lib.configs()]
        else: cfg=[dict(c,cid=c['cid']+1000,a=0.5) for c in lib.configs() if c['p']==30 and c['mis']==1]
        return df, pd.DataFrame(cfg)
    res=pickle.load(open(f'results_{which}.pkl','rb'))
    rows=[]; cfg=[]
    for c,a,rr in res:
        cfg.append(dict(c,a=a)); rows+=rr
    return pd.DataFrame(rows), pd.DataFrame(cfg)
def holm(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p); adj=np.empty(m); run=0
    for k,i in enumerate(o):
        run=max(run,(m-k)*p[i]); adj[i]=min(1,run)
    return adj
def paired(df,metric,a,b):
    x=df[df.method==a].set_index(['cid','r'])[metric]; y=df[df.method==b].set_index(['cid','r'])[metric]
    out=[]
    for cid,g in x.groupby(level=0):
        xx=g.dropna(); yy=y.loc[cid].reindex(xx.index.get_level_values(1))
        d=(xx.values-yy.values); d=d[~np.isnan(d)]
        if len(d)<10: continue
        mref=np.nanmean(yy.values); md=d.mean(); se=d.std(ddof=1)/np.sqrt(len(d))
        t=md/se if se>0 else 0.0; p=2*stats.t.sf(abs(t),len(d)-1) if se>0 else 1.0
        q=stats.t.ppf(.975,len(d)-1)
        out.append(dict(cid=cid,method=a,ref=b,metric=metric,n=len(d),mean_diff=md,ci_lo=md-q*se,ci_hi=md+q*se,
                        rel_pct=100*md/mref,rel_lo=100*(md-q*se)/mref,rel_hi=100*(md+q*se)/mref,t=t,p=p))
    return out
def families(df,cfg):
    F={}
    r=[];[r.extend(paired(df,'test_mse',m,'OLS')) for m in FEAS]; F['F1_testMSE_vs_OLS']=pd.DataFrame(r)
    r=[];[r.extend(paired(df,'se_b0',m,'OLS')) for m in FEAS]; F['F2_coefSE_vs_OLS']=pd.DataFrame(r)
    r=[]
    for m in ORIG:
        for ref in ['CV-power','CV-linear']:
            for met in ['test_mse','se_b0']: r.extend(paired(df,met,m,ref))
    F['F3_orig_vs_CV']=pd.DataFrame(r)
    for k,v in F.items():
        v['p_holm']=holm(v.p.values); F[k]=v.merge(cfg[['cid','p','T','rho','v','mis']],on='cid')
    return F
if __name__=='__main__':
    df,cfg=load('main')
    if has_pickle('main'): df.to_csv('replication_level_main.csv.gz',index=False)
    summ=df.groupby(['cid','method']).agg(test_mse=('test_mse','mean'),se_b0=('se_b0','mean'),se_bS=('se_bS','mean'),
        cov_err=('cov_err','mean'),prec_err=('prec_err','mean'),time=('time','mean'),intensity=('intensity','mean'),
        n_ok=('test_mse','count')).reset_index().merge(cfg,on='cid')
    summ.to_csv('summary_main.csv',index=False)
    notes=df[df.note.fillna('').str.startswith('failed')]; print('failures',len(notes))
    F=families(df,cfg)
    for k,v in F.items():
        v.to_csv(f'{k}.csv',index=False)
        print(k,len(v),'sig(p_holm<.05):',int((v.p_holm<.05).sum()))
