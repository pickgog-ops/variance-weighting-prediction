"""Paired tests of the benchmark comparison (50 outer folds per data set).

Methods: A = OLS, B = OLS+Scale, C = FGLS, D = TWLS, and the standardised Ridge, Lasso and ENet.
Contrasts per data set: each of Ridge, Lasso, ENet, C and D against A, and D against C; for CRPS
and NLL also B against A, C against B and D against B. This gives families of 30 (MSE),
45 (CRPS) and 45 (NLL) tests, each Holm-adjusted separately.

Each test is the Nadeau-Bengio corrected resampled t test on the 50 paired fold losses
(difference = candidate - reference, variance factor 1/50 + 0.2/0.8, 49 df);
the 95% intervals are pointwise.

Output: results/tests_benchmarks_holm.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from _paths import stage_root
ROOT = stage_root()
RES = ROOT / "results"
DS = ["Auto MPG", "Insurance", "Hitters", "Engel", "Abalone"]


def nb_test(d, test_frac=0.2):
    d = np.asarray(d, float); J = len(d)
    se = np.sqrt(1.0 / J + test_frac / (1 - test_frac)) * d.std(ddof=1)
    if se == 0 or not np.isfinite(se):
        return 0.0, 1.0, d.mean(), d.mean(), se
    t = d.mean() / se
    half = stats.t.ppf(.975, J - 1) * se
    return t, 2 * stats.t.sf(abs(t), J - 1), d.mean() - half, d.mean() + half, se


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


folds = pd.read_csv(RES / "raw_scoring_folds.csv")
folds["split_id"] = folds["repeat"] * 5 + folds["fold"]
std = pd.read_csv(RES / "raw_std_penalized.csv")
comp = pd.read_csv(ROOT / "reference" / "realdata_reference_metrics.csv", encoding="utf-8-sig")
comp = comp[comp.protocol == "cv50"]


def losses(ds, method, metric, restored):
    """Fold losses ordered by split_id. C and D use the original losses, except in
    contrasts with B, where all three use the learned-scale losses of the reference file."""
    if method in ("Ridge", "Lasso", "ElasticNet"):
        return std[(std.dataset == ds) & (std.method == method + "-std")].sort_values("split_id")[metric].to_numpy()
    if method == "A":
        return folds[folds.dataset == ds].sort_values("split_id")[f"OLS_{metric}"].to_numpy()
    if method in ("C", "D") and not restored:
        col = {"C": "WLS-ML", "D": "TWLS"}[method]
        return folds[folds.dataset == ds].sort_values("split_id")[f"{col}_{metric}"].to_numpy()
    return comp[(comp.dataset == ds) & (comp.group == method)].sort_values("split_id")[metric].to_numpy()


rows = []
for metric in ("mse", "crps", "nll"):
    for ds in DS:
        pairs = [(m, "A", False) for m in ("Ridge", "Lasso", "ElasticNet", "C", "D")]
        if metric != "mse":
            pairs += [("B", "A", True), ("C", "B", True), ("D", "B", True)]
        pairs += [("D", "C", True)]
        for cand, ref, restored in pairs:
            a, b = losses(ds, ref, metric, restored), losses(ds, cand, metric, restored)
            d = b - a
            t, p, lo, hi, se = nb_test(d)
            rows.append(dict(metric=metric, dataset=ds, contrast=f"{cand}-{ref}", n=len(d),
                             reference_mean=a.mean(), candidate_mean=b.mean(), difference_mean=d.mean(),
                             corrected_se=se, t_statistic=t, p_raw=p, ci95_low=lo, ci95_high=hi))
R = pd.DataFrame(rows)
R["p_holm"] = np.nan
for m in ("mse", "crps", "nll"):
    idx = R.metric == m
    R.loc[idx, "p_holm"] = holm(R.loc[idx, "p_raw"].to_numpy())
R["family_size"] = R.groupby("metric")["metric"].transform("size")
R.to_csv(RES / "tests_benchmarks_holm.csv", index=False)
print(R[R.p_holm <= 0.05][["metric", "dataset", "contrast", "difference_mean", "p_holm"]].to_string(index=False))
