"""
run_experiments.py
==================
Reproduces every quantitative result reported in the manuscript.

Convention
----------
Legacy descriptive experiments use repeated 70/30 splits. The primary
comparison and the variance/iteration sensitivity analyses use repeated
five-fold cross-validation, matching the manuscript.

Outputs (CSV) -> ../results/
    tab1_datasets.csv        dataset summary
    tab2_hetero.csv          Breusch-Pagan / White tests + diagnostics
    tab3_main.csv            point-prediction accuracy (PRIMARY TABLE)
    tab4_varmodel.csv        variance-model complexity ablation
    tab5_iterations.csv      convergence vs number of alternating updates
    tab6_intervals.csv       prediction-interval quality (PICP / PINAW)
    tab6b_condcov.csv        conditional coverage by noise tercile
    tab7_bootstrap.csv       bootstrap standard errors of coefficients
    tab8_runtime.csv         training wall-clock time
    tabS1_crossval.csv       5x5-fold cross-validation
    raw_main_splits.csv      per-split MSEs underlying Table 3
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RepeatedKFold

from wls_lib import (load_all_datasets, IterativeWLS, make_baseline,
                     train_test_split_idx, mse, select_temper,
                     heteroscedasticity_tests, picp, pinaw,
                     conditional_coverage, VARIANCE_MODELS, TEMPER_GRID,
                     RANDOM_SEED, RESULTS_DIR)

N_REPEAT = 50
KEY = {"Linear": "Linear", "Poly2": "Poly2", "Poly3": "Poly3",
       "RandomForest": "RF"}
BASELINES = ["OLS", "Ridge", "Lasso", "ElasticNet"]


# --------------------------------------------------------------------------- #
def curvature_gain(X, y):
    """% reduction in in-sample MSE when a full quadratic mean model replaces
    the linear one.  Proxy for mean-function misspecification."""
    lin = LinearRegression().fit(X, y)
    Z = PolynomialFeatures(2, include_bias=False).fit_transform(X)
    quad = LinearRegression().fit(Z, y)
    m_lin = np.mean((y - lin.predict(X)) ** 2)
    m_quad = np.mean((y - quad.predict(Z)) ** 2)
    return (m_lin - m_quad) / m_lin * 100.0


# --------------------------------------------------------------------------- #
def table1(datasets):
    return pd.DataFrame([d.summary() for d in datasets.values()])


def table2(datasets):
    rows = []
    for name, v in datasets.items():
        t = heteroscedasticity_tests(v.X, v.y)
        rows.append(dict(
            dataset=name, n=v.n,
            bp_stat=t["bp_stat"], bp_p=t["bp_p"],
            white_stat=t["white_stat"], white_p=t["white_p"],
            var_ratio=t["var_ratio"],
            curvature_gain_pct=curvature_gain(v.X, v.y),
            reject_h0_5pct="yes" if min(t["bp_p"], t["white_p"]) < 0.05 else "no",
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def table3(datasets):
    """PRIMARY RESULT.  50 repeated 70/30 splits."""
    rows, raw = [], []
    for name, v in datasets.items():
        rec = {m: [] for m in BASELINES + ["WLS-ML", "TWLS"]}
        gammas = []
        for s in range(N_REPEAT):
            tr, te = train_test_split_idx(v.n, seed=s)
            Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
            for m in BASELINES:
                rec[m].append(mse(yte, make_baseline(m, s).fit(Xtr, ytr).predict(Xte)))
            rec["WLS-ML"].append(
                mse(yte, IterativeWLS("Linear", 5, temper=1.0, seed=s).fit(Xtr, ytr).predict(Xte)))
            g, _ = select_temper(Xtr, ytr, "Linear", seed=s)
            gammas.append(g)
            rec["TWLS"].append(
                mse(yte, IterativeWLS("Linear", 5, temper=g, seed=s).fit(Xtr, ytr).predict(Xte)))
            raw.append(dict(dataset=name, seed=s, gamma=g,
                            **{m: rec[m][-1] for m in rec}))

        ols = np.array(rec["OLS"])
        row = dict(dataset=name, n=v.n, mean_gamma=float(np.mean(gammas)))
        for m in rec:
            a = np.array(rec[m])
            row[f"{m}_mean"] = a.mean()
            row[f"{m}_sd"] = a.std(ddof=1)
            row[f"{m}_impr_pct"] = float(np.mean((ols - a) / ols * 100))
            if m != "OLS":
                try:
                    row[f"{m}_wilcoxon_p"] = float(
                        stats.wilcoxon(ols, a, alternative="two-sided").pvalue)
                    row[f"{m}_win_rate"] = float(np.mean(a < ols) * 100)
                except ValueError:
                    row[f"{m}_wilcoxon_p"] = np.nan
                    row[f"{m}_win_rate"] = np.nan
        rows.append(row)
        print(f"  [tab3] {name} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(raw)


# --------------------------------------------------------------------------- #
def table4(datasets):
    """Variance-model ablation under 5-fold x 10-repeat cross-validation."""
    rows = []
    for name, v in datasets.items():
        rec = {KEY[k]: [] for k in VARIANCE_MODELS}
        rec["OLS"] = []
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
        for s, (tr, te) in enumerate(rkf.split(v.X)):
            Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
            rec["OLS"].append(mse(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)))
            for k in VARIANCE_MODELS:
                rec[KEY[k]].append(
                    mse(yte, IterativeWLS(k, 5, temper=1.0, seed=s).fit(Xtr, ytr).predict(Xte)))
        ols = np.array(rec["OLS"])
        row = dict(dataset=name)
        for m in ["OLS"] + [KEY[k] for k in VARIANCE_MODELS]:
            a = np.array(rec[m])
            row[f"{m}_mean"] = a.mean()
            row[f"{m}_fold_impr_pct"] = float(np.mean((ols - a) / ols * 100))
            row[f"{m}_impr_pct"] = float((ols.mean() - a.mean()) / ols.mean() * 100)
            row[f"{m}_win_rate"] = float(np.mean(a < ols) * 100)
        rows.append(row)
        print(f"  [tab4] {name} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def table5(datasets, grid=(0, 1, 2, 3, 5, 8, 10)):
    """Iteration sensitivity under 5-fold x 4-repeat cross-validation."""
    rows = []
    for name, v in datasets.items():
        row = dict(dataset=name)
        for T in grid:
            vals = []
            rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=42)
            for s, (tr, te) in enumerate(rkf.split(v.X)):
                m = (LinearRegression() if T == 0
                     else IterativeWLS("Linear", T, temper=1.0, seed=s))
                vals.append(mse(v.y[te], m.fit(v.X[tr], v.y[tr]).predict(v.X[te])))
            row[f"T{T}_mean"] = float(np.mean(vals))
            row[f"T{T}_impr_pct"] = np.nan  # filled in below once the T = 0 baseline is known
        base = row["T0_mean"]
        for T in grid:
            row[f"T{T}_impr_pct"] = (base - row[f"T{T}_mean"]) / base * 100
        rows.append(row)
        print(f"  [tab5] {name} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def table6(datasets, alpha=0.05, n_seeds=20):
    """Prediction intervals: constant OLS vs adaptive nested-CV TWLS."""
    z = stats.norm.ppf(1 - alpha / 2)
    rows, cond = [], []
    for name, v in datasets.items():
        agg = {k: [] for k in ["OLS_PICP", "OLS_PINAW", "WLS_PICP", "WLS_PINAW",
                               "OLS_cov_dev", "WLS_cov_dev"]}
        for s in range(n_seeds):
            tr, te = train_test_split_idx(v.n, seed=s)
            Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
            ols = LinearRegression().fit(Xtr, ytr)
            sd = np.std(ytr - ols.predict(Xtr), ddof=1)
            p = ols.predict(Xte)
            lo, hi = p - z * sd, p + z * sd
            agg["OLS_PICP"].append(picp(yte, lo, hi))
            agg["OLS_PINAW"].append(pinaw(yte, lo, hi))
            agg["OLS_cov_dev"].append(abs(picp(yte, lo, hi) - (1 - alpha) * 100))

            g, _ = select_temper(Xtr, ytr, "Linear", seed=s)
            w = IterativeWLS("Linear", 5, temper=g, seed=s).fit(Xtr, ytr)
            s2 = w.predict_sigma2(Xte) * np.exp(1.2704)   # E[log chi2_1] correction
            sh = np.sqrt(np.maximum(s2, 1e-12))
            pw = w.predict(Xte)
            lo, hi = pw - z * sh, pw + z * sh
            agg["WLS_PICP"].append(picp(yte, lo, hi))
            agg["WLS_PINAW"].append(pinaw(yte, lo, hi))
            agg["WLS_cov_dev"].append(abs(picp(yte, lo, hi) - (1 - alpha) * 100))

            # Both methods use the TWLS-predicted noise strata, on all seeds.
            cc_w = conditional_coverage(yte, lo, hi, sh, 3)
            cc_w["dataset"], cc_w["method"] = name, "WLS (adaptive)"
            lo2, hi2 = p - z * sd, p + z * sd
            cc_o = conditional_coverage(yte, lo2, hi2, sh, 3)
            cc_o["dataset"], cc_o["method"] = name, "OLS (constant)"
            cond.extend([cc_w, cc_o])

        rows.append(dict(
            dataset=name,
            OLS_PICP=float(np.mean(agg["OLS_PICP"])),
            OLS_PINAW=float(np.mean(agg["OLS_PINAW"])),
            OLS_cov_dev=float(np.mean(agg["OLS_cov_dev"])),
            WLS_PICP=float(np.mean(agg["WLS_PICP"])),
            WLS_PINAW=float(np.mean(agg["WLS_PINAW"])),
            WLS_cov_dev=float(np.mean(agg["WLS_cov_dev"])),
            PINAW_reduction_pct=float(
                (np.mean(agg["OLS_PINAW"]) - np.mean(agg["WLS_PINAW"]))
                / np.mean(agg["OLS_PINAW"]) * 100),
        ))
        print(f"  [tab6] {name} done", flush=True)
    cc = pd.concat(cond, ignore_index=True)
    cc['sigma_tercile'] = cc['group'].str.extract(r'\((low|mid|high) sigma\)', expand=False)
    cc = cc.groupby(["dataset", "sigma_tercile", "method"], observed=True,
                    as_index=False)[["coverage_pct", "mean_width"]].mean()
    return pd.DataFrame(rows), cc


# --------------------------------------------------------------------------- #
def table7(datasets, B=1000):
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for name, v in datasets.items():
        X, y, n = v.X, v.y, v.n
        boot = {m: [] for m in ["OLS", "WLS"]}
        pt = {}
        for m in ["OLS", "WLS"]:
            est = (LinearRegression() if m == "OLS"
                   else IterativeWLS("Linear", 5, temper=1.0))
            est.fit(X, y)
            pt[m] = (np.r_[est.intercept_, est.coef_] if m == "OLS"
                     else est.coef_.copy())
        for b in range(B):
            idx = rng.integers(0, n, n)
            Xb, yb = X[idx], y[idx]
            if np.linalg.matrix_rank(np.column_stack([np.ones(n), Xb])) < Xb.shape[1] + 1:
                continue
            boot["OLS"].append(np.r_[LinearRegression().fit(Xb, yb).intercept_,
                                     LinearRegression().fit(Xb, yb).coef_])
            boot["WLS"].append(IterativeWLS("Linear", 5, temper=1.0).fit(Xb, yb).coef_)
        se = {m: np.std(np.array(boot[m]), axis=0, ddof=1) for m in boot}
        names = ["intercept"] + v.feature_names
        for j, nm in enumerate(names):
            rows.append(dict(
                dataset=name, coefficient=nm,
                OLS_estimate=pt["OLS"][j], OLS_se=se["OLS"][j],
                WLS_estimate=pt["WLS"][j], WLS_se=se["WLS"][j],
                se_reduction_pct=(se["OLS"][j] - se["WLS"][j]) / se["OLS"][j] * 100,
            ))
        print(f"  [tab7] {name} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def table8(datasets):
    rows = []
    for name, v in datasets.items():
        tr, _ = train_test_split_idx(v.n, seed=RANDOM_SEED)
        Xtr, ytr = v.X[tr], v.y[tr]
        for m in BASELINES + ["WLS-ML", "TWLS"]:
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                if m == "TWLS":
                    select_temper(Xtr, ytr, "Linear", seed=RANDOM_SEED)
                elif m in BASELINES:
                    make_baseline(m).fit(Xtr, ytr)
                else:
                    IterativeWLS("Linear", 5, temper=1.0).fit(Xtr, ytr)
                ts.append(time.perf_counter() - t0)
            rows.append(dict(dataset=name, method=m,
                             train_time_ms=float(np.median(ts) * 1000)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def tableS1(datasets, n_splits=5, n_repeats=5):
    from sklearn.model_selection import KFold
    rows = []
    for name, v in datasets.items():
        for r in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True,
                       random_state=RANDOM_SEED + r)
            rec = {m: [] for m in BASELINES + ["WLS-ML", "TWLS"]}
            for tr, te in kf.split(v.X):
                Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
                for m in BASELINES:
                    rec[m].append(mse(yte, make_baseline(m).fit(Xtr, ytr).predict(Xte)))
                rec["WLS-ML"].append(mse(
                    yte, IterativeWLS("Linear", 5, temper=1.0).fit(Xtr, ytr).predict(Xte)))
                g, _ = select_temper(Xtr, ytr, "Linear")
                rec["TWLS"].append(mse(
                    yte, IterativeWLS("Linear", 5, temper=g).fit(Xtr, ytr).predict(Xte)))
            ols = np.mean(rec["OLS"])
            for m in rec:
                rows.append(dict(dataset=name, repeat=r, method=m,
                                 cv_mse_mean=float(np.mean(rec[m])),
                                 cv_mse_sd=float(np.std(rec[m], ddof=1)),
                                 impr_vs_OLS_pct=(ols - np.mean(rec[m])) / ols * 100))
        print(f"  [tabS1] {name} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ds = load_all_datasets()
    t1 = table1(ds);            t1.to_csv(f"{RESULTS_DIR}/tab1_datasets.csv", index=False)
    print("tab1 ok")
    t2 = table2(ds);            t2.to_csv(f"{RESULTS_DIR}/tab2_hetero.csv", index=False)
    print("tab2 ok")
    t3, raw = table3(ds)
    t3.to_csv(f"{RESULTS_DIR}/tab3_main.csv", index=False)
    raw.to_csv(f"{RESULTS_DIR}/raw_main_splits.csv", index=False)
    print("tab3 ok")
    t4 = table4(ds);            t4.to_csv(f"{RESULTS_DIR}/tab4_varmodel.csv", index=False)
    print("tab4 ok")
    t5 = table5(ds);            t5.to_csv(f"{RESULTS_DIR}/tab5_iterations.csv", index=False)
    print("tab5 ok")
    t6, t6b = table6(ds)
    t6.to_csv(f"{RESULTS_DIR}/tab6_intervals.csv", index=False)
    t6b.to_csv(f"{RESULTS_DIR}/tab6b_condcov.csv", index=False)
    print("tab6 ok")
    t7 = table7(ds);            t7.to_csv(f"{RESULTS_DIR}/tab7_bootstrap.csv", index=False)
    print("tab7 ok")
    t8 = table8(ds);            t8.to_csv(f"{RESULTS_DIR}/tab8_runtime.csv", index=False)
    print("tab8 ok")
    s1 = tableS1(ds);           s1.to_csv(f"{RESULTS_DIR}/tabS1_crossval.csv", index=False)
    print("tabS1 ok")

    pd.set_option("display.width", 250)
    print("\n=== Table 2 (heteroscedasticity) ===")
    print(t2.round(4).to_string(index=False))
    print("\n=== Table 3 (main) ===")
    print(t3.round(4).to_string(index=False))
    print("\n=== Table 4 (variance model) ===")
    print(t4.round(4).to_string(index=False))
    print("\n=== Prediction intervals ===")
    print(t6.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
