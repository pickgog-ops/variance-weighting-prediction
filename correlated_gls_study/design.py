import numpy as np
BETA=np.array([1.0,1.0])
def make_sigma(p,rho,spread):
    j=np.arange(p)
    R=rho**np.abs(j[:,None]-j[None,:])
    d=np.exp(np.linspace(-0.5,0.5,p)*np.log(spread)) if spread>1 else np.ones(p)
    d=d/np.mean(d)                      # mean variance 1
    S=np.sqrt(d)[:,None]*R*np.sqrt(d)[None,:]
    return S
def unit_means(p): return np.linspace(-1,1,p)
def targets(p,Sigma,a):
    """Exact unweighted (beta0) and Sigma^{-1}-weighted (betaS) population
    projections for y_tj = 1 + x_tj + a (x_tj^2 - 1) + e_tj, x_tj ~ N(mu_j,1) indep."""
    mu=unit_means(p)
    Ez=np.column_stack([np.ones(p),mu])                 # E z_j
    Ezz=np.array([[[1,m],[m,m*m+1]] for m in mu])        # E z_j z_j'
    Em=a*(mu**2)                                         # E m_j = a(mu^2+1-1)
    Ezm=np.column_stack([Em, a*(mu**3+3*mu-mu)])         # E[z_j m_j]: x(x^2-1) -> mu^3+3mu - mu
    def proj(A):
        G=np.zeros((2,2)); h=np.zeros(2)
        for i in range(p):
            for j in range(p):
                if i==j: G+=A[i,i]*Ezz[i]; h+=A[i,i]*Ezm[i]
                else: G+=A[i,j]*np.outer(Ez[i],Ez[j]); h+=A[i,j]*Ez[i]*Em[j]
        return BETA+np.linalg.solve(G,h), G
    b0,G0=proj(np.eye(p)); bS,_=proj(np.linalg.inv(Sigma))
    excess=(bS-b0)@G0@(bS-b0)/p      # per-point excess unweighted risk of the GLS target
    return b0,bS,excess
if __name__=='__main__':
    for p in (10,30,60):
        for rho in (0.3,0.8):
            for sp in (1,20):
                S=make_sigma(p,rho,sp); b0,bS,ex=targets(p,S,0.5)
                print(p,rho,sp,'cond %.1f'%np.linalg.cond(S),'b0',b0.round(3),'bS',bS.round(3),'excess/pt %.4f'%ex)
