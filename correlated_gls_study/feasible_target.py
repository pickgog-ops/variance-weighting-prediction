"""Probability limit of one-step FGLS when Sigma-hat -> E[e e'] (residual second
moment) under the omitted quadratic term.  Added after the verification of the main results
to correct the target-shift interpretation."""
import numpy as np, pandas as pd
from design import make_sigma, unit_means, targets, BETA

def proj(p, A, a):
    mu = unit_means(p); Ez = np.column_stack([np.ones(p), mu])
    Ezz = np.array([[[1, m], [m, m * m + 1]] for m in mu])
    Em = a * mu ** 2
    Ezm = np.column_stack([Em, a * (mu ** 3 + 2 * mu)])
    G = np.zeros((2, 2)); h = np.zeros(2)
    for i in range(p):
        for j in range(p):
            if i == j:
                G += A[i, i] * Ezz[i]; h += A[i, i] * Ezm[i]
            else:
                G += A[i, j] * np.outer(Ez[i], Ez[j]); h += A[i, j] * Ez[i] * Em[j]
    return BETA + np.linalg.solve(G, h), G

def resid_second_moment(p, Sigma, a, b0):
    """Sigma + E[r r'], r_j = a(x_j^2-1) - (b0-beta)'z_j, x_j ~ N(mu_j,1), independent across j."""
    mu = unit_means(p); d0, d1 = b0 - BETA
    Er = a * mu ** 2 - d0 - d1 * mu
    var = a * a * (4 * mu ** 2 + 2) + d1 * d1 - 2 * a * d1 * (2 * mu)
    M = np.outer(Er, Er); M[np.diag_indices(p)] += var
    return Sigma + M

def feasible_targets(p, rho, v, a, Sigma=None):
    Sig = make_sigma(p, rho, v) if Sigma is None else Sigma
    b0, bS, _ = targets(p, Sig, a)
    St = resid_second_moment(p, Sig, a, b0)
    bF, _ = proj(p, np.linalg.inv(St), a)
    bD, _ = proj(p, np.diag(1 / np.diag(St)), a)
    _, G = proj(p, np.eye(p), a)
    ex = lambda b: (b - b0) @ G @ (b - b0) / p
    return dict(b0=b0, bS=bS, bF=bF, bD=bD, ex_S=ex(bS), ex_F=ex(bF), ex_D=ex(bD))

if __name__ == '__main__':
    rows = []
    for a in (0.25, 0.5):
        for p in (10, 30, 60):
            for rho in (0.3, 0.8):
                for v in (1, 20):
                    if a == 0.5 and p != 30:
                        continue
                    t = feasible_targets(p, rho, v, a)
                    rows.append(dict(a=a, p=p, rho=rho, v=v, b0_slope=t['b0'][1], bS_slope=t['bS'][1],
                                     bF_slope=t['bF'][1], bDiag_slope=t['bD'][1],
                                     excess_S=t['ex_S'], excess_F=t['ex_F'], excess_Diag=t['ex_D']))
    R = pd.DataFrame(rows); R.to_csv('feasible_targets.csv', index=False)
    print(R.round(4).to_string())
