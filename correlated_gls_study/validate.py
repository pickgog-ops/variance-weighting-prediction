import numpy as np, pandas as pd, lib, warnings
from sklearn.covariance import LedoitWolf
from nonlinshrink_port import fsopt, loss_mv
rng=np.random.default_rng(1)
out=[]
# V1 LW official vs sklearn
d1=[]
for _ in range(20):
    p=int(rng.integers(5,80)); n=int(rng.integers(15,300)); E=rng.standard_normal((n,p))@np.diag(rng.uniform(.5,3,p))
    d1.append(np.max(np.abs(lib.est_lw(E)-LedoitWolf(assume_centered=True).fit(E).covariance_)))
out.append(('V1 LW official cov1Para(k=0) vs sklearn LedoitWolf', max(d1)))
# V2 OAS Chen formula vs sklearn
d2=[]
for _ in range(20):
    p=int(rng.integers(5,80)); n=int(rng.integers(15,300)); E=rng.standard_normal((n,p))@np.diag(rng.uniform(.5,3,p))
    d2.append(np.max(np.abs(lib.est_oas(E)-lib.est_oas_sklearn(E))))
out.append(('V2 OAS Chen et al. (2010) Eq.23 vs sklearn OAS', max(d2)))
# V3 nonlinshrink unit test
x=np.array([[-0.98511153,-0.2599713],[0.20374114,-0.59699234],[0.28570754,0.77542166],[-0.78768299,0.45808448],[0.80494623,-2.16308943],[-2.01761751,-0.27944538],[-0.24710646,-0.33260174],[0.369508,0.57123535],[2.07913481,-0.0749686],[1.07194672,-0.69343179],[-0.11947645,-1.45457297],[-0.54877366,-1.21191041]])
exp=np.array([[1.04344299,0.0335051],[0.0335051,1.11112703]])
src=open('nonlinshrink_port.py').read().replace('assert n >= 12, "sample size n must be >= 12"','pass')
ns={}; exec(compile(src,'port_noassert','exec'),ns)
out.append(('V3 LW-NL port vs package unit-test values (k=0; size assertion bypassed because the test has n=12)', np.max(np.abs(ns['shrink_cov'](x,k=0)-exp))))
# V4 PRIAL under minimum-variance loss, LW (2020) design: eigenvalues 1 (20%), 3 (40%), 10 (40%)
for p,n in [(100,200),(200,100)]:
    ev=np.r_[np.ones(p//5),3*np.ones(2*p//5),10*np.ones(p-p//5-2*p//5)]; Sig=np.diag(ev)
    res={k:[] for k in ['LW-linear','OAS','LW-NL','QIS']}; base=[]; orc=[]
    for rep in range(30):
        Y=rng.standard_normal((n,p))*np.sqrt(ev)
        S=Y.T@Y/n
        F=fsopt(S,Sig)
        lo=loss_mv(F,Sig); orc.append(lo)
        ls=loss_mv(S,Sig) if p<n else np.nan; base.append(ls)
        for k,fn in [('LW-linear',lib.est_lw),('OAS',lib.est_oas),('LW-NL',lib.est_lwnl),('QIS',lib.est_qis)]:
            res[k].append(loss_mv(fn(Y),Sig))
    if p<n:
        for k in res:
            prial=(np.mean(base)-np.mean(res[k]))/(np.mean(base)-np.mean(orc))*100
            out.append((f'V4 PRIAL(MV) % p={p} n={n} {k}',prial))
    else:
        for k in res: out.append((f'V4 MV loss p={p} n={n} {k} (oracle {np.mean(orc):.4f})',np.mean(res[k])))
# V5 exact targets vs large-sample estimates
from design import make_sigma,targets
p=10; Sig=make_sigma(p,0.8,20); b0,bS,_=targets(p,Sig,0.25)
Z,y=lib.gen(np.random.default_rng(7),400000,p,Sig,0.25)
out.append(('V5 exact beta0 vs OLS(T=4e5) max diff',np.max(np.abs(lib.gls(Z,y,None)-b0))))
out.append(('V5 exact betaSigma vs oracle GLS(T=4e5) max diff',np.max(np.abs(lib.gls(Z,y,np.linalg.inv(Sig))-bS))))
# V6 oracle GLS sampling covariance vs theory (correct mean)
p=10;T=40; Sig=make_sigma(p,0.8,20); W=np.linalg.inv(Sig); bs=[];Gs=[]
for rep in range(4000):
    Z,y=lib.gen(np.random.default_rng(1000+rep),T,p,Sig,0.0); bs.append(lib.gls(Z,y,W)); Gs.append(np.einsum('tjk,ij,til->kl',Z,W,Z))
theo=np.linalg.inv(np.mean(Gs,0)); emp=np.cov(np.array(bs).T)
out.append(('V6 oracle GLS: empirical/theoretical slope variance ratio (approx, Jensen)',emp[1,1]/theo[1,1]))
for k,v in out: print(f'{k}: {v:.6g}')
