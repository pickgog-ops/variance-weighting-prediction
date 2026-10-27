"""Numbers of the main-text benchmark and simulation tables, from the saved results.

  MSE table       OLS MSE; % MSE change vs OLS (positive = lower loss) for the
                  standardised Ridge, Lasso and ENet, OLS+Scale, FGLS and TWLS; mean gamma
  CRPS table      % CRPS change vs OLS for the same methods
  Interval table  PICP and PINAW of A: OLS, B: OLS+Scale, C: FGLS, D: TWLS; % PINAW change of D vs A
  Simulation      mean gamma, % MSE change and % SD reduction of the slope (FGLS, TWLS), S1-S4

OLS+Scale keeps the OLS mean and adds the learned variance function; its losses and the
FGLS/TWLS interval columns with learned scale are read from reference/realdata_reference_metrics.csv
(groups B, C and D; "cv50" = the 50 outer folds, "holdout20" = the 20 interval splits).

Output: results/main_text_values.csv (table, row, column, value)
"""
import os
import pandas as pd

from _paths import stage_root
ROOT = stage_root()
RES = ROOT / "results"
DS = ["Auto MPG", "Insurance", "Hitters", "Engel", "Abalone"]
LAB = {"Ridge": "Ridge", "Lasso": "Lasso", "ElasticNet": "ENet"}

cv = pd.read_csv(RES / "tab3_main_cv.csv").set_index("dataset")
sc = pd.read_csv(RES / "tab_scoring.csv").set_index("dataset")
folds = pd.read_csv(RES / "raw_scoring_folds.csv")
std = pd.read_csv(RES / "raw_std_penalized.csv")
iv = pd.read_csv(RES / "tab6_intervals.csv").set_index("dataset")
sim = pd.read_csv(RES / "tab_sim.csv").set_index("scenario")
comp = pd.read_csv(ROOT / "reference" / "realdata_reference_metrics.csv", encoding="utf-8-sig")
c50 = comp[comp.protocol == "cv50"].groupby(["dataset", "group"])[["mse", "crps"]].mean()
h20 = comp[comp.protocol == "holdout20"].groupby(["dataset", "group"])[["picp", "pinaw"]].mean()

out = []
add = lambda t, r, c, v: out.append(dict(table=t, row=r, column=c, value=float(v)))
for d in DS:
    add("MSE", d, "OLS MSE", cv.loc[d, "OLS_mse"])
    for metric, table in (("mse", "MSE"), ("crps", "CRPS")):
        base = folds[folds.dataset == d][f"OLS_{metric}"].mean()
        for m in ("Ridge", "Lasso", "ElasticNet"):
            v = std[(std.dataset == d) & (std.method == m + "-std")][metric].mean()
            add(table, d, LAB[m], 100 * (base - v) / base)
        add(table, d, "OLS+Scale", 100 * (base - c50.loc[(d, "B"), metric]) / base)
    add("MSE", d, "FGLS", cv.loc[d, "WLS-ML_impr_pct"])
    add("MSE", d, "TWLS", cv.loc[d, "TWLS_impr_pct"])
    add("MSE", d, "mean gamma", cv.loc[d, "mean_gamma"])
    add("CRPS", d, "FGLS", sc.loc[d, "WLS-ML_dCRPS_pct"])
    add("CRPS", d, "TWLS", sc.loc[d, "TWLS_dCRPS_pct"])
    add("Intervals", d, "A PICP", iv.loc[d, "OLS_PICP"]); add("Intervals", d, "A PINAW", iv.loc[d, "OLS_PINAW"])
    for g in ("B", "C"):
        add("Intervals", d, f"{g} PICP", h20.loc[(d, g), "picp"]); add("Intervals", d, f"{g} PINAW", h20.loc[(d, g), "pinaw"])
    add("Intervals", d, "D PICP", iv.loc[d, "WLS_PICP"]); add("Intervals", d, "D PINAW", iv.loc[d, "WLS_PINAW"])
    add("Intervals", d, "PINAW change D vs A", iv.loc[d, "PINAW_reduction_pct"])
for s, lab in (("strong", "S1"), ("mild", "S2"), ("none", "S3"), ("strong+misspec", "S4")):
    add("Simulation", lab, "mean gamma", sim.loc[s, "mean_gamma"])
    add("Simulation", lab, "FGLS MSE change", sim.loc[s, "WLS-ML_impr_pct"])
    add("Simulation", lab, "TWLS MSE change", sim.loc[s, "TWLS_impr_pct"])
    add("Simulation", lab, "FGLS SD reduction", sim.loc[s, "WLS-ML_eff_gain_pct"])
    add("Simulation", lab, "TWLS SD reduction", sim.loc[s, "TWLS_eff_gain_pct"])
pd.DataFrame(out).to_csv(RES / "main_text_values.csv", index=False)
o = pd.DataFrame(out)
for t in o.table.unique():
    print(f"\n{t}\n" + o[o.table == t].pivot_table(index="row", columns="column", values="value", sort=False).round(3).to_string())
