"""Simulation for the variance-shrinkage rules: S1-S4 x n in {200,500,2000}, 200 reps.
Data generation follows exp_simulation.py
(seed rule default_rng(1000 r), d=2, test size 20,000)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import c123_lib as L
W = L.W

from _paths import stage_root
ROOT = str(stage_root())
D, N_TEST, N_REP = 2, 20000, 200
SCEN = {"S1": ("strong", False), "S2": ("mild", False), "S3": ("none", False), "S4": ("strong", True)}
FINE = tuple(np.round(np.arange(0, 1.0001, 0.05), 2))


def make_dgp(het, misspec, d=D):  # identical to exp_simulation.make_dgp
    beta = np.r_[1.0, np.linspace(1.0, -1.0, d)]
    def mean_fn(X):
        m = np.column_stack([np.ones(len(X)), X]) @ beta
        if misspec:
            m = m + 1.5 * (X[:, 0] ** 2 - 1.0)
        return m
    g = {"strong": np.r_[0.0, np.full(d, 1.6)], "mild": np.r_[0.0, np.full(d, 0.5)]}.get(het)
    def var_fn(X):
        if g is None:
            return np.ones(len(X))
        return np.exp(np.column_stack([np.ones(len(X)), X]) @ g)
    return mean_fn, var_fn, beta, g


def weighted_target(mean_fn, var_fn, n=2_000_000, seed=987654):
    """Population WLS target beta(W) with true inverse-variance weights."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, D)); A = np.column_stack([np.ones(n), X])
    w = 1.0 / var_fn(X)
    return np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * mean_fn(X)))


def main(n_list):
    rows = []
    targets = {}
    for sname, (het, mis) in SCEN.items():
        mean_fn, var_fn, beta, _ = make_dgp(het, mis)
        targets[sname] = (beta.copy(), weighted_target(mean_fn, var_fn))
    pd.DataFrame([dict(scenario=s, beta0=list(np.round(b0, 6)), betaW=list(np.round(bw, 6)))
                  for s, (b0, bw) in targets.items()]).to_csv(
        os.path.join(ROOT, "results", "sim_targets.csv"), index=False)
    for n in n_list:
        for sname, (het, mis) in SCEN.items():
            mean_fn, var_fn, beta, _ = make_dgp(het, mis)
            b0, bW = targets[sname]
            t_start = time.perf_counter()
            for r in range(N_REP):
                rng = np.random.default_rng(1000 * r)
                Xtr = rng.normal(0, 1, (n, D))
                ytr = mean_fn(Xtr) + rng.normal(0, 1, n) * np.sqrt(var_fn(Xtr))
                Xte = rng.normal(0, 1, (N_TEST, D))
                yte = mean_fn(Xte) + rng.normal(0, 1, N_TEST) * np.sqrt(var_fn(Xte))
                fits, inten, secs = {}, {}, {}
                def timed(key, fn):
                    t0 = time.perf_counter(); out = fn(); secs[key] = time.perf_counter() - t0; return out
                fits["OLS"] = timed("OLS", lambda: L.AMV(L.map_power(0.0)).fit(Xtr, ytr)); inten["OLS"] = 0.0
                fits["FGLS"] = timed("FGLS", lambda: L.AMV(L.map_power(1.0)).fit(Xtr, ytr)); inten["FGLS"] = 1.0
                def cvt():
                    g, _ = L.cv_select(Xtr, ytr, L.map_power, seed=r)
                    return g, L.AMV(L.map_power(g)).fit(Xtr, ytr)
                inten["CV-TWLS"], fits["CV-TWLS"] = timed("CV-TWLS", cvt)
                fits["EB-power"] = timed("EB-power", lambda: L.AMV(L.map_eb_power).fit(Xtr, ytr))
                fits["LW-linear"] = timed("LW-linear", lambda: L.AMV(L.map_lw_linear).fit(Xtr, ytr))
                def cvl():
                    p, _ = L.cv_select(Xtr, ytr, L.map_linear, seed=r)
                    return p, L.AMV(L.map_linear(p)).fit(Xtr, ytr)
                inten["CV-linear"], fits["CV-linear"] = timed("CV-linear", cvl)
                fits["ALS"] = timed("ALS", lambda: L.ALS().fit(Xtr, ytr))
                for k in ("EB-power", "LW-linear", "ALS"):
                    inten[k] = fits[k].intensity_
                # descriptive oracle envelopes (selected on the test set; not implementable)
                for fam, fac in (("Oracle-power", L.map_power), ("Oracle-linear", L.map_linear)):
                    best = None
                    for g in FINE:
                        m = L.AMV(fac(g)).fit(Xtr, ytr)
                        e = W.mse(yte, m.predict(Xte))
                        if best is None or e < best[0]:
                            best = (e, g, m)
                    fits[fam], inten[fam], secs[fam] = best[2], best[1], np.nan
                for m, f in fits.items():
                    c = f.coef_
                    rows.append(dict(scenario=sname, n_train=n, replication=r, method=m,
                                     test_mse=W.mse(yte, f.predict(Xte)),
                                     excess_risk=float(np.sum((c - b0) ** 2)),
                                     dist_betaW=float(np.sum((c - bW) ** 2)),
                                     beta0_hat=c[0], beta1_hat=c[1], beta2_hat=c[2],
                                     intensity=inten[m], fit_seconds=secs[m]))
            print(f"done {sname} n={n} {time.perf_counter()-t_start:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", f"raw_C123_sim_partial.csv"), index=False)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "raw_C123_simulation.csv"), index=False)


if __name__ == "__main__":
    main([int(v) for v in sys.argv[1:]] if len(sys.argv) > 1 else [500, 200, 2000])
