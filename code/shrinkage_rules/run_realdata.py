"""Benchmark part of the variance-shrinkage comparison: the same 50 outer folds (10 x 5-fold CV) as the main study."""
import json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import c123_lib as L
W = L.W

from _paths import stage_root
ROOT = str(stage_root())
ref = pd.read_csv(os.path.join(ROOT, "reference", "realdata_reference_metrics.csv"), encoding="utf-8-sig")
ref = ref[ref.protocol == "cv50"]
datasets = W.load_all_datasets(os.path.join(ROOT, "data"))

rows = []
for name, ds in datasets.items():
    pairs = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42).split(ds.X)
    for sid, (tr, te) in enumerate(pairs):
        r = ref[(ref.dataset == name) & (ref.split_id == sid)].set_index("group")
        gamma = float(r.loc["D", "group_gamma"])
        assert int(r.loc["D", "n_train"]) == len(tr) and int(r.loc["D", "n_test"]) == len(te)
        Xtr, ytr, Xte, yte = ds.X[tr], ds.y[tr], ds.X[te], ds.y[te]
        fits = {}
        t0 = time.perf_counter(); fits["OLS"] = L.AMV(L.map_power(0.0)).fit(Xtr, ytr); tO = time.perf_counter() - t0
        t0 = time.perf_counter(); fits["FGLS"] = L.AMV(L.map_power(1.0)).fit(Xtr, ytr); tF = time.perf_counter() - t0
        # CV-TWLS: archived inner-CV gamma (reference group D); its timing is re-measured with a fresh
        # inner search on the same inner folds and the selection is checked against the reference file.
        t0 = time.perf_counter(); g_new, _ = L.cv_select(Xtr, ytr, L.map_power, seed=42)
        fits["CV-TWLS"] = L.AMV(L.map_power(gamma)).fit(Xtr, ytr); tT = time.perf_counter() - t0
        t0 = time.perf_counter(); fits["EB-power"] = L.AMV(L.map_eb_power).fit(Xtr, ytr); tE = time.perf_counter() - t0
        t0 = time.perf_counter(); fits["LW-linear"] = L.AMV(L.map_lw_linear).fit(Xtr, ytr); tL = time.perf_counter() - t0
        t0 = time.perf_counter(); rho_cv, _ = L.cv_select(Xtr, ytr, L.map_linear, seed=42)
        fits["CV-linear"] = L.AMV(L.map_linear(rho_cv)).fit(Xtr, ytr); tC = time.perf_counter() - t0
        t0 = time.perf_counter(); fits["ALS"] = L.ALS().fit(Xtr, ytr); tA = time.perf_counter() - t0
        times = dict(zip(["OLS", "FGLS", "CV-TWLS", "EB-power", "LW-linear", "CV-linear", "ALS"],
                         [tO, tF, tT, tE, tL, tC, tA]))
        inten = {"OLS": 0.0, "FGLS": 1.0, "CV-TWLS": gamma, "CV-linear": rho_cv,
                 "EB-power": fits["EB-power"].intensity_, "LW-linear": fits["LW-linear"].intensity_,
                 "ALS": fits["ALS"].intensity_}
        for m, f in fits.items():
            mse, crps, nll = L.gaussian_scores(yte, f.predict(Xte), f.predict_sigma2(Xte))
            rows.append(dict(dataset=name, split_id=sid, method=m, mse=mse, crps=crps, nll=nll,
                             intensity=inten[m], gamma_archived=gamma, gamma_refit=g_new,
                             fit_seconds=times[m], n_train=len(tr), n_test=len(te)))
    print("done", name, flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results", "raw_C123_realdata.csv"), index=False)

# ---- implementation check against the reference metrics ------------------ #
checks = []
for m, g in (("OLS", "B"), ("FGLS", "C"), ("CV-TWLS", "D")):
    for metric in ("mse", "crps", "nll"):
        a = out[out.method == m].sort_values(["dataset", "split_id"])[metric].to_numpy()
        b = ref[ref.group == g].sort_values(["dataset", "split_id"])[metric].to_numpy()
        checks.append(dict(method=m, ref_group=g, metric=metric, n=len(a),
                           max_abs_diff=float(np.max(np.abs(a - b))),
                           max_rel_diff=float(np.max(np.abs(a - b) / np.abs(b)))))
g = out[out.method == "CV-TWLS"]
checks.append(dict(method="CV-TWLS", ref_group="D", metric="gamma_reselected_equals_archived",
                   n=len(g), max_abs_diff=float(np.max(np.abs(g.gamma_refit - g.gamma_archived))),
                   max_rel_diff=float(np.mean(g.gamma_refit == g.gamma_archived))))
pd.DataFrame(checks).to_csv(os.path.join(ROOT, "results", "check_C123_realdata_vs_reference.csv"), index=False)
print(pd.DataFrame(checks).to_string())
