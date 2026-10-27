"""
exp_simulation.py
=================
Monte-Carlo study with a KNOWN data-generating process.

Purpose
-------
Real-data benchmarks cannot separate two confounded explanations for why
feasible WLS may fail to improve test MSE:

  (i)   finite-sample noise in sigma-hat^2(x), and
  (ii)  misspecification of the MEAN function (the residual scale pattern then
        reflects curvature, not noise, and inverse-variance weighting
        amplifies the resulting specification bias).

With a controlled DGP we can switch (ii) on and off at will and evaluate
against a 20,000-point test set, so that the expected MSE is measured almost
exactly.

Scenarios
---------
  S1  well-specified mean, STRONG heteroscedasticity
  S2  well-specified mean, MILD  heteroscedasticity
  S3  well-specified mean, HOMOSCEDASTIC  (safety check -> no harm)
  S4  MISSPECIFIED mean (omitted quadratic term), strong heteroscedasticity

Output -> results/tab_sim.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LinearRegression

from wls_lib import (IterativeWLS, make_baseline, mse, select_temper,
                     RESULTS_DIR)

N_TRAIN = 500
N_TEST = 20000
D = 2
N_REP = 200
BASELINES = ["OLS", "Ridge", "Lasso", "ElasticNet"]


def make_dgp(het: str, misspec: bool, d: int = D):
    """Return (mean_fn, var_fn) for the requested scenario."""
    beta = np.r_[1.0, np.linspace(1.0, -1.0, d)]

    def mean_fn(X):
        m = np.column_stack([np.ones(len(X)), X]) @ beta
        if misspec:                       # omitted curvature
            m = m + 1.5 * (X[:, 0] ** 2 - 1.0)
        return m

    if het == "strong":
        g = np.r_[0.0, np.full(d, 1.6)]
    elif het == "mild":
        g = np.r_[0.0, np.full(d, 0.5)]
    else:                                  # homoscedastic
        g = None

    def var_fn(X):
        if g is None:
            return np.ones(len(X))
        return np.exp(np.column_stack([np.ones(len(X)), X]) @ g)

    return mean_fn, var_fn, beta


def run_scenario(het, misspec, n_train=N_TRAIN, d=D, n_rep=N_REP, seed0=0):
    mean_fn, var_fn, beta = make_dgp(het, misspec, d)
    meths = BASELINES + ["WLS-ML", "TWLS"]
    rec = {m: [] for m in meths}
    coefs = {m: [] for m in ["OLS", "WLS-ML", "TWLS"]}
    gammas = []

    for r in range(n_rep):
        rng = np.random.default_rng(seed0 + 1000 * r)
        Xtr = rng.normal(0, 1, (n_train, d))
        ytr = mean_fn(Xtr) + rng.normal(0, 1, n_train) * np.sqrt(var_fn(Xtr))
        Xte = rng.normal(0, 1, (N_TEST, d))
        yte = mean_fn(Xte) + rng.normal(0, 1, N_TEST) * np.sqrt(var_fn(Xte))

        for m in BASELINES:
            rec[m].append(mse(yte, make_baseline(m, r).fit(Xtr, ytr).predict(Xte)))
        w = IterativeWLS("Linear", 5, temper=1.0, seed=r).fit(Xtr, ytr)
        rec["WLS-ML"].append(mse(yte, w.predict(Xte)))
        g, _ = select_temper(Xtr, ytr, "Linear", seed=r)
        gammas.append(g)
        wt = IterativeWLS("Linear", 5, temper=g, seed=r).fit(Xtr, ytr)
        rec["TWLS"].append(mse(yte, wt.predict(Xte)))

        coefs["OLS"].append(np.r_[LinearRegression().fit(Xtr, ytr).intercept_,
                                  LinearRegression().fit(Xtr, ytr).coef_])
        coefs["WLS-ML"].append(w.coef_)
        coefs["TWLS"].append(wt.coef_)

    ols = np.array(rec["OLS"])
    row = dict(scenario=f"{het}{'+misspec' if misspec else ''}",
               mean_gamma=float(np.mean(gammas)))
    for m in meths:
        a = np.array(rec[m])
        row[f"{m}_mse"] = a.mean()
        row[f"{m}_impr_pct"] = (ols.mean() - a.mean()) / ols.mean() * 100
        row[f"{m}_win"] = float(np.mean(a < ols) * 100)
        row[f"{m}_p"] = float(stats.wilcoxon(ols, a).pvalue) if m != "OLS" else np.nan
    # estimation efficiency: MC standard deviation of the slope on x1
    for m in ["OLS", "WLS-ML", "TWLS"]:
        C = np.array(coefs[m])
        row[f"{m}_se_beta1"] = float(C[:, 1].std(ddof=1))
        if m != "OLS":
            row[f"{m}_eff_gain_pct"] = (np.array(coefs["OLS"])[:, 1].std(ddof=1)
                                        - C[:, 1].std(ddof=1)) \
                / np.array(coefs["OLS"])[:, 1].std(ddof=1) * 100
    return row


def main():
    scen = [("strong", False), ("mild", False), ("none", False), ("strong", True)]
    rows = [run_s := run_scenario(h, m) for h, m in scen]
    for r in rows:
        print("done", r["scenario"], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS_DIR}/tab_sim.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["scenario", "mean_gamma"] + \
           [f"{m}_impr_pct" for m in BASELINES[1:] + ["WLS-ML", "TWLS"]] + \
           [f"{m}_p" for m in ["WLS-ML", "TWLS"]] + \
           [f"{m}_eff_gain_pct" for m in ["WLS-ML", "TWLS"]]
    print(out[[c for c in cols if c in out.columns]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
