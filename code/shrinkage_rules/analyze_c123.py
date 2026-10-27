"""Summaries and paired tests for the variance-shrinkage rules (simulation and benchmarks)."""
import os, json
import numpy as np
import pandas as pd
from scipy import stats

from _paths import stage_root
ROOT = str(stage_root())
RES = os.path.join(ROOT, "results")
NEW = ["EB-power", "LW-linear", "CV-linear", "ALS"]
REFS = ["CV-TWLS", "OLS"]


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------- simulation ------------------------------- #
sim = pd.read_csv(os.path.join(RES, "raw_C123_simulation.csv"))
assert len(sim) == 3 * 4 * 200 * 9
# implementation check against the reference replications (n=500)
rec = pd.read_csv(os.path.join(ROOT, "reference", "simulation_reference_replications.csv"))
smap = {"strong": "S1", "mild": "S2", "none": "S3", "strong+misspec": "S4"}
rec["scenario"] = rec.scenario.map(smap)
chk = []
s5 = sim[sim.n_train == 500]
for ours, theirs in (("OLS", "OLS"), ("FGLS", "WLS-ML"), ("CV-TWLS", "TWLS")):
    a = s5[s5.method == ours].sort_values(["scenario", "replication"])
    b = rec.sort_values(["scenario", "replication"])
    chk.append(dict(method=ours, quantity="test_mse", max_abs_diff=float(np.max(np.abs(a.test_mse.values - b[f"{theirs}_mse"].values)))))
    chk.append(dict(method=ours, quantity="beta1", max_abs_diff=float(np.max(np.abs(a.beta1_hat.values - b[f"{theirs}_beta1"].values)))))
a = s5[s5.method == "CV-TWLS"].sort_values(["scenario", "replication"])
chk.append(dict(method="CV-TWLS", quantity="gamma", max_abs_diff=float(np.max(np.abs(a.intensity.values - b.gamma.values)))))
pd.DataFrame(chk).to_csv(os.path.join(RES, "check_C123_sim_vs_reference.csv"), index=False)

summ = []
for (s, n, m), g in sim.groupby(["scenario", "n_train", "method"]):
    ols = sim[(sim.scenario == s) & (sim.n_train == n) & (sim.method == "OLS")].sort_values("replication")
    g = g.sort_values("replication")
    summ.append(dict(scenario=s, n_train=n, method=m, mean_test_mse=g.test_mse.mean(),
                     rel_to_OLS_pct=100 * (ols.test_mse.mean() - g.test_mse.mean()) / ols.test_mse.mean(),
                     mean_excess_risk=g.excess_risk.mean(), mean_dist_betaW=g.dist_betaW.mean(),
                     beta1_bias=g.beta1_hat.mean() - 1.0, beta1_sd=g.beta1_hat.std(ddof=1),
                     beta1_rmse=float(np.sqrt(np.mean((g.beta1_hat - 1.0) ** 2))),
                     intensity_mean=g.intensity.mean(), intensity_sd=g.intensity.std(ddof=1),
                     intensity_q10=g.intensity.quantile(.1), intensity_median=g.intensity.median(),
                     intensity_q90=g.intensity.quantile(.9),
                     frac_intensity_0=float((g.intensity == 0).mean()),
                     frac_intensity_1=float((g.intensity == 1).mean()),
                     median_fit_seconds=g.fit_seconds.median()))
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(RES, "summary_C123_simulation.csv"), index=False)

tests = []
for (s, n), g in sim.groupby(["scenario", "n_train"]):
    piv = g.pivot(index="replication", columns="method", values="test_mse")
    for m in NEW:
        for r in REFS:
            d = piv[m] - piv[r]
            sd = d.std(ddof=1)
            if sd == 0:
                t, p = 0.0, 1.0
            else:
                t = d.mean() / (sd / np.sqrt(len(d))); p = 2 * stats.t.sf(abs(t), len(d) - 1)
            half = stats.t.ppf(.975, len(d) - 1) * sd / np.sqrt(len(d))
            tests.append(dict(scenario=s, n_train=n, method=m, reference=r, n=len(d),
                              mean_diff=d.mean(), rel_diff_pct_of_ref=100 * d.mean() / piv[r].mean(),
                              ci_low=d.mean() - half, ci_high=d.mean() + half, t=t, p_raw=p,
                              n_lower=int((d < 0).sum()), n_higher=int((d > 0).sum())))
tests = pd.DataFrame(tests)
tests["p_holm"] = holm(tests.p_raw)
assert len(tests) == 96
tests["decision"] = np.where(tests.p_holm < .05, np.where(tests.mean_diff < 0, "lower risk", "higher risk"), "unresolved")
tests.to_csv(os.path.join(RES, "tests_C123_simulation_holm96.csv"), index=False)

# ------------------------------- real data -------------------------------- #
rd = pd.read_csv(os.path.join(RES, "raw_C123_realdata.csv"))
assert len(rd) == 5 * 50 * 7
rsum = []
for (ds, m), g in rd.groupby(["dataset", "method"]):
    rsum.append(dict(dataset=ds, method=m, mse=g.mse.mean(), crps=g.crps.mean(), nll=g.nll.mean(),
                     intensity_mean=g.intensity.mean(), intensity_median=g.intensity.median(),
                     frac_intensity_0=float((g.intensity == 0).mean()),
                     frac_intensity_1=float((g.intensity == 1).mean()),
                     median_fit_seconds=g.fit_seconds.median()))
pd.DataFrame(rsum).to_csv(os.path.join(RES, "summary_C123_realdata.csv"), index=False)
rt = []
for metric in ("mse", "crps", "nll"):
    block = []
    for ds, g in rd.groupby("dataset"):
        piv = g.pivot(index="split_id", columns="method", values=metric)
        for m in NEW:
            for r in REFS:
                d = (piv[m] - piv[r]).to_numpy()
                sd = d.std(ddof=1)
                se = np.sqrt((1 / 50 + .2 / .8)) * sd
                if se == 0:
                    t, p = 0.0, 1.0
                else:
                    t = d.mean() / se; p = 2 * stats.t.sf(abs(t), 49)
                half = stats.t.ppf(.975, 49) * se
                block.append(dict(metric=metric, dataset=ds, method=m, reference=r, mean_diff=d.mean(),
                                  rel_diff_pct_of_ref=100 * d.mean() / piv[r].mean() if metric != "nll" else np.nan,
                                  ci_low=d.mean() - half, ci_high=d.mean() + half, t=t, p_raw=p,
                                  n_lower=int((d < 0).sum()), n_higher=int((d > 0).sum())))
    block = pd.DataFrame(block); block["p_holm"] = holm(block.p_raw); assert len(block) == 40
    rt.append(block)
rt = pd.concat(rt)
rt["decision"] = np.where(rt.p_holm < .05, np.where(rt.mean_diff < 0, "lower loss", "higher loss"), "unresolved")
rt.to_csv(os.path.join(RES, "tests_C123_realdata_holm40.csv"), index=False)
print("analysis complete")
