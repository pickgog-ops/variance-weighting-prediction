"""POST HOC exploratory check (not in the pre-specified protocol), added after the
verification of the main results: does the misspecified-mean harm depend on the
variance being aligned with the covariate means?  Variance arrangements:
  reversed : variances decrease in the unit index (anti-aligned with mu_j)
  permuted : variances assigned to units by a fixed random permutation (unaligned)
p = 30, v = 20, omitted curvature a = 0.25, T in {20,40,100,400}, rho in {0.3,0.8},
200 paired replications each.  lib.py is not modified; make_sigma is wrapped here."""
import pickle, time, sys
import numpy as np
from multiprocessing import Pool
import lib, design

_orig = design.make_sigma
PERM = np.random.default_rng(424242).permutation(30)

def make_sigma_arranged(p, rho, v, order):
    j = np.arange(p)
    R = rho ** np.abs(j[:, None] - j[None, :])
    d = np.exp(np.linspace(-0.5, 0.5, p) * np.log(v)) if v > 1 else np.ones(p)
    d = d / np.mean(d)
    if order == 'reversed':
        d = d[::-1]
    elif order == 'permuted':
        d = d[PERM]
    return np.sqrt(d)[:, None] * R * np.sqrt(d)[None, :]

def job(args):
    cfg, order = args
    lib.make_sigma = lambda p, rho, v: make_sigma_arranged(p, rho, v, order)
    rows = []
    for r in range(200):
        for row in lib.one_rep(cfg, r):
            row['order'] = order
            rows.append(row)
    return cfg, order, rows

if __name__ == '__main__':
    tasks = []
    cid = 2000
    for order in ('reversed', 'permuted'):
        for T in (20, 40, 100, 400):
            for rho in (0.3, 0.8):
                tasks.append((dict(cid=cid, p=30, T=T, rho=rho, v=20, mis=1), order))
                cid += 1
    t = time.time(); res = []
    with Pool(2) as pool:
        for i, out in enumerate(pool.imap_unordered(job, tasks)):
            res.append(out)
            print(i, len(tasks), '%.0fs' % (time.time() - t), flush=True)
    pickle.dump(res, open('results_posthoc_alignment.pkl', 'wb'))
    print('done')
