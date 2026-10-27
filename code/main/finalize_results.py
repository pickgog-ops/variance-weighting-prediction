"""
finalize_results.py
===================
Produces the definitive result tables used in the manuscript:

  tab3_main_cv.csv     PRIMARY point-prediction table under repeated 5-fold CV
                       (5 folds x 10 repeats = 50 paired evaluations)
  tab6b_condcov.csv    conditional coverage of 95% intervals, stratified by the
                       estimated noise level (fixed grouping for OLS and WLS)
  tab_sim_sweep.csv    Monte-Carlo sweep over the degree of mean misspecification

Outputs also refresh: tab6_intervals.csv (20-seed average, unchanged protocol)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

from wls_lib import (load_all_datasets, IterativeWLS, make_baseline, mse,
                     select_temper, picp, pinaw, RANDOM_SEED, RESULTS_DIR)

BASELINES = ["OLS", "Ridge", "Lasso", "ElasticNet"]
METHODS = BASELINES + ["WLS-ML", "TWLS"]
N_SPLITS, N_REPEATS = 5, 10


def corrected_resampled_test(reference, candidate, test_fraction=0.2):
    """Nadeau--Bengio corrected resampled t test for paired losses.

    Positive differences mean that ``candidate`` has lower loss.  The usual
    fold-level Wilcoxon test treats overlapping training sets as independent
    and is therefore anti-conservative for repeated cross-validation.
    """
    d = np.asarray(reference, float) - np.asarray(candidate, float)
    correction = 1.0 / len(d) + test_fraction / (1.0 - test_fraction)
    se = np.sqrt(correction * np.var(d, ddof=1))
    if not np.isfinite(se) or se == 0:
        return np.nan, np.nan, np.nan, np.nan
    stat = float(np.mean(d) / se)
    p = float(2.0 * stats.t.sf(abs(stat), df=len(d) - 1))
    q = float(stats.t.ppf(0.975, df=len(d) - 1))
    return stat, p, float(np.mean(d) - q * se), float(np.mean(d) + q * se)


# --------------------------------------------------------------------------- #
def tab3_main_cv(datasets):
    rows, raw = [], []
    for name, v in datasets.items():
        rec = {m: [] for m in METHODS}
        gammas = []
        rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
        for k, (tr, te) in enumerate(rkf.split(v.X)):
            Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
            for m in BASELINES:
                rec[m].append(mse(yte, make_baseline(m, k).fit(Xtr, ytr).predict(Xte)))
            rec["WLS-ML"].append(
                mse(yte, IterativeWLS("Linear", 5, temper=1.0).fit(Xtr, ytr).predict(Xte)))
            g, _ = select_temper(Xtr, ytr, "Linear")
            gammas.append(g)
            rec["TWLS"].append(
                mse(yte, IterativeWLS("Linear", 5, temper=g).fit(Xtr, ytr).predict(Xte)))
            raw.append(dict(dataset=name, repeat=k // N_SPLITS,
                            fold=k % N_SPLITS, gamma=g,
                            **{m: rec[m][-1] for m in METHODS}))

        ols = np.array(rec["OLS"])
        row = dict(dataset=name, n=v.n, mean_gamma=float(np.mean(gammas)),
                   median_gamma=float(np.median(gammas)))
        for m in METHODS:
            a = np.array(rec[m])
            row[f"{m}_mse"] = a.mean()
            row[f"{m}_sd"] = a.std(ddof=1)
            row[f"{m}_impr_pct"] = (ols.mean() - a.mean()) / ols.mean() * 100
            if m != "OLS":
                row[f"{m}_win_rate"] = float(np.mean(a < ols) * 100)
                row[f"{m}_wilcoxon_p"] = float(stats.wilcoxon(ols, a).pvalue)
                stat, p, lo, hi = corrected_resampled_test(ols, a)
                row[f"{m}_corr_t"] = stat
                row[f"{m}_p"] = p
                row[f"{m}_diff_ci_lo"] = lo
                row[f"{m}_diff_ci_hi"] = hi
        rows.append(row)
        print(f"  [cv] {name}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(raw)


# --------------------------------------------------------------------------- #
def tab_conditional_coverage(datasets, alpha=0.05, n_seeds=20):
    """Stratify coverage by the WLS-estimated noise level so that the OLS
    constant-width interval and the WLS adaptive interval are compared on
    identical strata."""
    z = stats.norm.ppf(1 - alpha / 2)
    out = []
    for name, v in datasets.items():
        agg = {}
        for s in range(n_seeds):
            tr, te = train_test_split_seed(v, s)
            Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
            ols = LinearRegression().fit(Xtr, ytr)
            sd = np.std(ytr - ols.predict(Xtr), ddof=1)
            p = ols.predict(Xte)
            g, _ = select_temper(Xtr, ytr, "Linear", seed=s)
            wls = IterativeWLS("Linear", 5, temper=g, seed=s).fit(Xtr, ytr)
            s2 = wls.predict_sigma2(Xte) * np.exp(1.2704)
            sh = np.sqrt(np.maximum(s2, 1e-12))
            pw = wls.predict(Xte)

            q = pd.qcut(sh, 3, labels=["low", "mid", "high"], duplicates="drop")
            for g in ["low", "mid", "high"]:
                m = np.asarray(q == g)
                if m.sum() == 0:
                    continue
                for lab, (lo, hi, wd) in {
                        "OLS (constant)": (p - z * sd, p + z * sd,
                                           np.full(len(yte), 2 * z * sd)),
                        "WLS (adaptive)": (pw - z * sh, pw + z * sh, 2 * z * sh),
                }.items():
                    key = (name, g, lab)
                    agg.setdefault(key, {"cov": [], "wid": []})
                    agg[key]["cov"].append(picp(yte[m], lo[m], hi[m]))
                    agg[key]["wid"].append(float(np.mean(wd[m])))
        for (dset, g, lab), d in agg.items():
            out.append(dict(dataset=dset, sigma_tercile=g, method=lab,
                            coverage_pct=float(np.mean(d["cov"])),
                            mean_width=float(np.mean(d["wid"]))))
    return pd.DataFrame(out)


def train_test_split_seed(v, s):
    from wls_lib import train_test_split_idx
    return train_test_split_idx(v.n, seed=s)


# --------------------------------------------------------------------------- #
def tab_sim_sweep(curv=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
                  n_train=500, d=2, n_rep=200):
    """Monte-Carlo: how does the benefit of reweighting erode as the mean model
    becomes progressively misspecified?"""
    rows = []
    for a in curv:
        rec = {"OLS": [], "WLS-ML": [], "TWLS": []}
        gs = []
        for r in range(n_rep):
            rng = np.random.default_rng(1000 * r + 7)
            beta = np.r_[1.0, np.linspace(1.0, -1.0, d)]
            g = np.r_[0.0, np.full(d, 1.6)]

            def mu(X):
                return np.column_stack([np.ones(len(X)), X]) @ beta + a * (X[:, 0] ** 2 - 1.0)

            def s2(X):
                return np.exp(np.column_stack([np.ones(len(X)), X]) @ g)

            Xtr = rng.normal(0, 1, (n_train, d))
            ytr = mu(Xtr) + rng.normal(0, 1, n_train) * np.sqrt(s2(Xtr))
            Xte = rng.normal(0, 1, (20000, d))
            yte = mu(Xte) + rng.normal(0, 1, 20000) * np.sqrt(s2(Xte))

            rec["OLS"].append(mse(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)))
            rec["WLS-ML"].append(
                mse(yte, IterativeWLS("Linear", 5, temper=1.0).fit(Xtr, ytr).predict(Xte)))
            gg, _ = select_temper(Xtr, ytr, "Linear")
            gs.append(gg)
            rec["TWLS"].append(
                mse(yte, IterativeWLS("Linear", 5, temper=gg).fit(Xtr, ytr).predict(Xte)))
        o = np.array(rec["OLS"])
        rows.append(dict(
            curvature=a,
            mean_gamma=float(np.mean(gs)),
            WLS_ML_impr_pct=(o.mean() - np.mean(rec["WLS-ML"])) / o.mean() * 100,
            TWLS_impr_pct=(o.mean() - np.mean(rec["TWLS"])) / o.mean() * 100,
            WLS_ML_p=float(stats.wilcoxon(o, np.array(rec["WLS-ML"])).pvalue),
            TWLS_p=float(stats.wilcoxon(o, np.array(rec["TWLS"])).pvalue),
        ))
        print(f"  [sim] curvature={a}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ds = load_all_datasets()
    t3, raw = tab3_main_cv(ds)
    t3.to_csv(f"{RESULTS_DIR}/tab3_main_cv.csv", index=False)
    raw.to_csv(f"{RESULTS_DIR}/raw_cv_folds.csv", index=False)

    cc = tab_conditional_coverage(ds)
    cc.to_csv(f"{RESULTS_DIR}/tab6b_condcov.csv", index=False)

    sw = tab_sim_sweep()
    sw.to_csv(f"{RESULTS_DIR}/tab_sim_sweep.csv", index=False)

    pd.set_option("display.width", 260)
    print("\n=== Tab3 (repeated 5-fold CV) ===")
    show = ["dataset", "mean_gamma"] + \
           [f"{m}_impr_pct" for m in METHODS[1:]] + \
           [f"{m}_p" for m in METHODS[1:]] + [f"{m}_win_rate" for m in METHODS[1:]]
    print(t3[show].round(4).to_string(index=False))
    print("\n=== Conditional coverage ===")
    print(cc.round(2).to_string(index=False))
    print("\n=== Simulation sweep ===")
    print(sw.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
