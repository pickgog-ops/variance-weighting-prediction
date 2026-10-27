"""
exp_scoring_rules.py
====================
The FGLS and CV-TWLS fits return a *predictive distribution*
    N( y_hat(x), sigma_hat^2(x) )
rather than a bare point prediction.  The proper way to compare such
probabilistic regressors is a strictly proper scoring rule, not MSE.

Metrics
-------
  NLL    Gaussian negative log-likelihood  0.5*log(2*pi*s2) + (y-yhat)^2/(2*s2)
  CRPS   continuous ranked probability score of the Gaussian predictive
  MSE    reported for completeness only (improper for a distributional forecast)

Baselines
---------
  OLS / Ridge / Lasso / ElasticNet  -> constant-variance Gaussian predictive
                                       (sigma^2 = training residual variance)
  WLS-ML, TWLS                      -> input-dependent Gaussian predictive

Protocol: repeated 5-fold CV, 5 splits x 10 repeats, paired by fold.

Output -> results/tab_scoring.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

from wls_lib import (load_all_datasets, IterativeWLS, make_baseline, mse,
                     select_temper, RANDOM_SEED, RESULTS_DIR)

BASELINES = ["OLS", "Ridge", "Lasso", "ElasticNet"]
METHODS = BASELINES + ["WLS-ML", "TWLS"]
N_SPLITS, N_REPEATS = 5, 10
LOG_CHI2 = 1.2704          # E[log chi^2_1] = psi(1/2) - log(1/2) = -1.2704


def corrected_resampled_p(reference, candidate, test_fraction=0.2):
    """Two-sided Nadeau--Bengio corrected resampled t-test p-value."""
    d = np.asarray(reference, float) - np.asarray(candidate, float)
    correction = 1.0 / len(d) + test_fraction / (1.0 - test_fraction)
    se = np.sqrt(correction * np.var(d, ddof=1))
    if not np.isfinite(se) or se == 0:
        return np.nan
    return float(2.0 * stats.t.sf(abs(np.mean(d) / se), df=len(d) - 1))


def gaussian_nll(y, mu, s2):
    s2 = np.maximum(s2, 1e-12)
    return float(np.mean(0.5 * np.log(2 * np.pi * s2) + (y - mu) ** 2 / (2 * s2)))


def gaussian_crps(y, mu, sd):
    """Closed-form CRPS of a Gaussian predictive distribution."""
    sd = np.maximum(sd, 1e-12)
    z = (y - mu) / sd
    return float(np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1)
                               + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))


def main():
    ds = load_all_datasets()
    rows, raw = [], []

    for name, v in ds.items():
        rec = {m: {"nll": [], "crps": [], "mse": []} for m in METHODS}
        rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                            random_state=RANDOM_SEED)
        for k, (tr, te) in enumerate(rkf.split(v.X)):
            Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]

            for m in BASELINES:
                est = make_baseline(m, k).fit(Xtr, ytr)
                mu = est.predict(Xte)
                s2 = np.var(ytr - est.predict(Xtr), ddof=1)
                rec[m]["nll"].append(gaussian_nll(yte, mu, np.full(len(yte), s2)))
                rec[m]["crps"].append(gaussian_crps(yte, mu, np.full(len(yte), np.sqrt(s2))))
                rec[m]["mse"].append(mse(yte, mu))

            for tag, temper in [("WLS-ML", 1.0),
                                ("TWLS", select_temper(Xtr, ytr, "Linear")[0])]:
                est = IterativeWLS("Linear", 5, temper=temper).fit(Xtr, ytr)
                mu = est.predict(Xte)
                s2 = est.predict_sigma2(Xte) * np.exp(LOG_CHI2)
                rec[tag]["nll"].append(gaussian_nll(yte, mu, s2))
                rec[tag]["crps"].append(gaussian_crps(yte, mu, np.sqrt(np.maximum(s2, 1e-12))))
                rec[tag]["mse"].append(mse(yte, mu))

            raw.append(dict(dataset=name, repeat=k // N_SPLITS,
                            fold=k % N_SPLITS,
                            **{f"{m}_{s}": rec[m][s][-1] for m in METHODS
                               for s in ("nll", "crps", "mse")}))

        row = dict(dataset=name, n=v.n)
        base_nll = np.array(rec["OLS"]["nll"])
        base_crps = np.array(rec["OLS"]["crps"])
        for m in METHODS:
            for s in ("nll", "crps", "mse"):
                a = np.array(rec[m][s])
                row[f"{m}_{s}"] = a.mean()
            row[f"{m}_dNLL"] = base_nll.mean() - np.array(rec[m]["nll"]).mean()
            row[f"{m}_dCRPS_pct"] = ((base_crps.mean() - np.array(rec[m]["crps"]).mean())
                                     / base_crps.mean() * 100)
            if m != "OLS":
                row[f"{m}_nll_wilcoxon_p"] = float(
                    stats.wilcoxon(base_nll, rec[m]["nll"]).pvalue)
                row[f"{m}_crps_wilcoxon_p"] = float(
                    stats.wilcoxon(base_crps, rec[m]["crps"]).pvalue)
                row[f"{m}_nll_p"] = corrected_resampled_p(base_nll, rec[m]["nll"])
                row[f"{m}_crps_p"] = corrected_resampled_p(base_crps, rec[m]["crps"])
                row[f"{m}_nll_win"] = float(np.mean(np.array(rec[m]["nll"]) < base_nll) * 100)
        rows.append(row)
        print(f"  [scoring] {name}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(f"{RESULTS_DIR}/tab_scoring.csv", index=False)
    pd.DataFrame(raw).to_csv(f"{RESULTS_DIR}/raw_scoring_folds.csv", index=False)

    pd.set_option("display.width", 280)
    print("\n=== NLL (lower is better; dNLL = OLS - method) ===")
    print(out[["dataset"] + [f"{m}_nll" for m in METHODS] +
              [f"{m}_dNLL" for m in METHODS[1:]] +
              [f"{m}_nll_p" for m in METHODS[1:]]].round(4).to_string(index=False))
    print("\n=== CRPS reduction vs OLS (%) ===")
    print(out[["dataset"] + [f"{m}_dCRPS_pct" for m in METHODS[1:]] +
              [f"{m}_crps_p" for m in METHODS[1:]]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
