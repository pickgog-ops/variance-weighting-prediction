"""Correlated-error GLS study: estimators, data generation and one replication.
See README.md.  All covariance estimators receive the same
training residual matrix E (T x p) from pooled OLS, without unit demeaning."""
import os, time
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(_v,"1")
import numpy as np, pandas as pd, warnings
from sklearn.covariance import OAS as SkOAS
from official_cov1para import cov1Para
from official_qis_eigh import QIS
from nonlinshrink_port import shrink_cov
from design import make_sigma, unit_means, targets, BETA

GRID=(0.0,0.25,0.5,0.75,1.0)
A_MIS=0.25
T_TEST=100
METHODS=['OLS','Oracle','Sample','Diag','LW-linear','OAS','LW-NL','QIS','CV-linear','CV-power']

# ---------------- data -----------------
def gen(rng,T,p,Sigma,a):
    mu=unit_means(p)
    x=mu[None,:]+rng.standard_normal((T,p))
    Z=np.stack([np.ones((T,p)),x],axis=2)              # T,p,2
    L=np.linalg.cholesky(Sigma)
    eps=rng.standard_normal((T,p))@L.T
    y=Z@BETA + a*(x**2-1) + eps
    return Z,y

# ---------------- regression -----------------
def gls(Z,y,W):
    """beta = (sum Z_t' W Z_t)^-1 sum Z_t' W y_t ; W = Sigma^-1 (or None for OLS)."""
    if W is None:
        G=np.einsum('tjk,tjl->kl',Z,Z); h=np.einsum('tjk,tj->k',Z,y)
    else:
        WZ=np.einsum('ij,tjk->tik',W,Z)
        G=np.einsum('tjk,tjl->kl',Z,WZ); h=np.einsum('tjk,tj->k',WZ,y)
    return np.linalg.solve(G,h)
def resid(Z,y,b): return y-Z@b

# ---------------- covariance estimators -----------------
def est_sample(E): return E.T@E/E.shape[0]
def est_diag(E): return np.diag(np.diag(est_sample(E)))
def est_lw(E): return np.asarray(cov1Para(pd.DataFrame(E),k=0),float)
def oas_chen(E):
    """Chen, Wiesel, Eldar and Hero (2010), Eq. (23), target tr(S)/p * I."""
    n,p=E.shape; S=est_sample(E)
    trS=np.trace(S); trS2=np.sum(S*S)
    num=(1-2/p)*trS2+trS**2
    den=(n+1-2/p)*(trS2-trS**2/p)
    rho=1.0 if den<=0 else min(num/den,1.0)
    return (1-rho)*S+rho*trS/p*np.eye(p), rho
def est_oas(E): return oas_chen(E)[0]
def est_oas_sklearn(E): return SkOAS(assume_centered=True).fit(E).covariance_
def est_lwnl(E): return shrink_cov(E,k=0)
def est_qis(E): return np.asarray(QIS(pd.DataFrame(E),k=0),float)

def path_inverse(S,kind,par,delta):
    lam,U=np.linalg.eigh(S); lam=np.clip(lam,0,None)
    if kind=='linear':
        ev=(1-par)*lam+par*lam.mean()+delta
    else:
        ev=(lam+delta)**par
        ev=ev*(lam.sum()/ev.sum())          # trace-preserving rescaling (does not affect GLS)
    Sig=(U*ev)@U.T; Inv=(U/ev)@U.T
    return Sig,Inv

def cv_select(Z,y,kind,rng,K=5):
    T=Z.shape[0]; idx=rng.permutation(T); folds=np.array_split(idx,K)
    err=np.zeros(len(GRID))
    for f in folds:
        tr=np.setdiff1d(idx,f)
        b0=gls(Z[tr],y[tr],None); E=resid(Z[tr],y[tr],b0); S=est_sample(E)
        delta=1e-6*np.trace(S)/S.shape[0]
        for g,par in enumerate(GRID):
            _,Inv=path_inverse(S,kind,par,delta)
            b=gls(Z[tr],y[tr],Inv)
            err[g]+=np.sum((y[f]-Z[f]@b)**2)
    return GRID[int(np.argmin(err))]          # first minimum in grid order

# ---------------- one replication -----------------
def one_rep(cfg,r,a_override=None):
    p,T,rho,v,mis,cid=cfg['p'],cfg['T'],cfg['rho'],cfg['v'],cfg['mis'],cfg['cid']
    a=(a_override if a_override is not None else A_MIS) if mis else 0.0
    seed=20261012+1000*cid+r
    rng=np.random.default_rng(seed)
    Sigma=make_sigma(p,rho,v); Sinv=np.linalg.inv(Sigma)
    b0,bS,_=targets(p,Sigma,a) if mis else (BETA,BETA,0.0)
    Z,y=gen(rng,T,p,Sigma,a); Zt,yt=gen(rng,T_TEST,p,Sigma,a)
    cv_rng=np.random.default_rng(seed+500)
    t0=time.perf_counter(); bols=gls(Z,y,None); t_ols=time.perf_counter()-t0
    E=resid(Z,y,bols)
    out=[]
    def record(m,b,SigHat,tt,inten=np.nan,note=''):
        row=dict(cid=cid,r=r,method=m,
                 test_mse=float(np.mean((yt-Zt@b)**2)),
                 se_b0=float(np.sum((b-b0)**2)), se_bS=float(np.sum((b-bS)**2)),
                 se_slope_b0=float((b[1]-b0[1])**2),
                 b_int=float(b[0]),b_slope=float(b[1]),time=tt,intensity=inten,note=note,
                 cov_err=np.nan,prec_err=np.nan)
        if SigHat is not None:
            row['cov_err']=float(np.linalg.norm(SigHat-Sigma)/np.linalg.norm(Sigma))
            try:
                if np.linalg.cond(SigHat)<1e14:
                    row['prec_err']=float(np.linalg.norm(np.linalg.inv(SigHat)-Sinv)/np.linalg.norm(Sinv))
            except np.linalg.LinAlgError: pass
        out.append(row)
    record('OLS',bols,None,t_ols)
    t0=time.perf_counter(); b=gls(Z,y,Sinv); record('Oracle',b,Sigma,time.perf_counter()-t0)
    for m,fn in [('Sample',est_sample),('Diag',est_diag),('LW-linear',est_lw),('OAS',est_oas),('LW-NL',est_lwnl),('QIS',est_qis)]:
        if m=='Sample' and T<=p:
            out.append(dict(cid=cid,r=r,method=m,note='undefined (T<=p)')); continue
        t0=time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                SH=fn(E); W=np.linalg.inv(SH); b=gls(Z,y,W)
            record(m,b,SH,t_ols+time.perf_counter()-t0)
        except Exception as ex:
            out.append(dict(cid=cid,r=r,method=m,note='failed: '+type(ex).__name__))
    for m,kind in [('CV-linear','linear'),('CV-power','power')]:
        t0=time.perf_counter()
        par=cv_select(Z,y,kind,cv_rng)
        S=est_sample(E); delta=1e-6*np.trace(S)/p
        SH,Inv=path_inverse(S,kind,par,delta); b=gls(Z,y,Inv)
        record(m,b,SH,t_ols+time.perf_counter()-t0,inten=par)
    return out

def configs():
    out=[]; cid=0
    for p in (10,30,60):
        for T in (20,40,100,400):
            for rho in (0.3,0.8):
                for v in (1,20):
                    for mis in (0,1):
                        out.append(dict(cid=cid,p=p,T=T,rho=rho,v=v,mis=mis)); cid+=1
    return out
