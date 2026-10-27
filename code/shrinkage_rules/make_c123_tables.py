"""Generate manuscript tables for the C123 intensity-rule comparison from saved results only."""
import os, sys
import numpy as np
import pandas as pd

RES = sys.argv[1]
OUT = sys.argv[2]
s = pd.read_csv(os.path.join(RES, "summary_C123_simulation.csv"))
t = pd.read_csv(os.path.join(RES, "tests_C123_simulation_holm96.csv"))
rs = pd.read_csv(os.path.join(RES, "summary_C123_realdata.csv"))
rt = pd.read_csv(os.path.join(RES, "tests_C123_realdata_holm40.csv"))

SC = {"S1": "S1", "S2": "S2", "S3": "S3", "S4": "S4"}
SIM_M = ["FGLS", "CV-TWLS", "CV-linear", "EB-power", "LW-linear", "ALS", "Oracle-power", "Oracle-linear"]
HDR = {"FGLS": "FGLS", "CV-TWLS": "CV-TWLS", "CV-linear": "CV-lin.", "EB-power": "EB-pow.",
       "LW-linear": "LW-lin.", "ALS": "ALS", "Oracle-power": "Or.-pow.", "Oracle-linear": "Or.-lin."}


def mark(sc, n, m):
    if m not in ("CV-linear", "EB-power", "LW-linear", "ALS"):
        return ""
    r = t[(t.scenario == sc) & (t.n_train == n) & (t.method == m) & (t.reference == "CV-TWLS")].iloc[0]
    if r.p_holm < 0.05:
        return "$^{\\dagger}$" if r.mean_diff < 0 else "$^{\\ddagger}$"
    return ""


def sim_rows(n, with_intensity):
    rows = []
    for sc in ["S1", "S2", "S3", "S4"]:
        g = s[(s.scenario == sc) & (s.n_train == n)].set_index("method")
        cells = []
        for m in SIM_M:
            v = f"{g.loc[m, 'rel_to_OLS_pct']:+.3f}" + mark(sc, n, m)
            cells.append(v)
        rows.append(f"{sc} & " + " & ".join(cells) + " \\\\")
        if with_intensity:
            ic = []
            for m in SIM_M:
                if m == "FGLS":
                    ic.append("")
                elif m == "ALS":
                    ic.append(f"({100*g.loc[m, 'intensity_mean']:.1f}\\%)")
                else:
                    ic.append(f"({g.loc[m, 'intensity_mean']:.2f})")
            rows.append(" & " + " & ".join(ic) + " \\\\")
    return rows


head = ("\\begin{tabular}{@{}l" + "r" * len(SIM_M) + "@{}}\n\\toprule\n"
        " & \\multicolumn{" + str(len(SIM_M)) + "}{c}{$\\Delta$ test MSE vs OLS (\\%); (mean selected intensity)} \\\\\n"
        "\\cmidrule(l){2-" + str(len(SIM_M) + 1) + "}\n"
        "Scen. & " + " & ".join(HDR[m] for m in SIM_M) + " \\\\\n\\midrule\n")
with open(os.path.join(OUT, "tab_C123_sim.tex"), "w") as f:
    f.write(head + "\n".join(sim_rows(500, True)) + "\n\\bottomrule\n\\end{tabular}\n")

full = head
for n in (200, 500, 2000):
    full += f"\\multicolumn{{{len(SIM_M)+1}}}{{@{{}}l}}{{\\emph{{n = {n}}}}} \\\\\n" + "\n".join(sim_rows(n, False)) + "\n"
    if n != 2000:
        full += "\\midrule\n"
with open(os.path.join(OUT, "tab_C123_sim_full.tex"), "w") as f:
    f.write(full + "\\bottomrule\n\\end{tabular}\n")

# Real data: MSE relative to OLS, with Holm marks vs CV-TWLS (dagger/ddagger) and vs OLS (*)
RM = ["FGLS", "CV-TWLS", "CV-linear", "EB-power", "LW-linear", "ALS"]
DS = ["Auto MPG", "Insurance", "Hitters", "Engel", "Abalone"]


def rmark(ds, m, metric):
    if m not in ("CV-linear", "EB-power", "LW-linear", "ALS"):
        return ""
    out = ""
    a = rt[(rt.metric == metric) & (rt.dataset == ds) & (rt.method == m)]
    r1 = a[a.reference == "CV-TWLS"].iloc[0]; r2 = a[a.reference == "OLS"].iloc[0]
    if r1.p_holm < 0.05:
        out += "$^{\\dagger}$" if r1.mean_diff < 0 else "$^{\\ddagger}$"
    if r2.p_holm < 0.05:
        out += "$^{*}$"
    return out


def real_table(metric, fname, fmt_rel=True):
    lines = ["\\begin{tabular}{@{}l" + "r" * len(RM) + "@{}}", "\\toprule",
             "Dataset & " + " & ".join(HDR[m] for m in RM) + " \\\\", "\\midrule"]
    for ds in DS:
        g = rs[rs.dataset == ds].set_index("method")
        base = g.loc["OLS", metric]
        cells = []
        for m in RM:
            if metric == "nll":
                v = f"{base - g.loc[m, metric]:+.3f}"
            else:
                v = f"{100*(base - g.loc[m, metric])/base:+.3f}"
            cells.append(v + rmark(ds, m, metric))
        lines.append(f"{ds} & " + " & ".join(cells) + " \\\\")
        if metric == "mse":
            ic = []
            for m in RM:
                if m == "FGLS":
                    ic.append("")
                elif m == "ALS":
                    ic.append(f"({100*g.loc[m, 'intensity_mean']:.0f}\\%)")
                else:
                    ic.append(f"({g.loc[m, 'intensity_mean']:.2f})")
            lines.append(" & " + " & ".join(ic) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, fname), "w") as f:
        f.write("\n".join(lines) + "\n")


real_table("mse", "tab_C123_real.tex")
real_table("crps", "tab_C123_real_crps.tex")
real_table("nll", "tab_C123_real_nll.tex")
print("tables written")

# Distribution of selected intensities (n = 500)
DM = [("CV-TWLS", "$\\hat\\gamma$ (CV)"), ("CV-linear", "$\\hat\\rho$ (CV)"),
      ("EB-power", "$\\hat\\kappa$ (EB)"), ("LW-linear", "$\\hat\\rho$ (LW)")]
lines = ["\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
         "Scen. & Intensity & Mean & 10\\% & Median & 90\\% & Share $=0$ & Share $=1$ \\\\", "\\midrule"]
for sc in ["S1", "S2", "S3", "S4"]:
    g = s[(s.scenario == sc) & (s.n_train == 500)].set_index("method")
    for k, (m, lab) in enumerate(DM):
        r = g.loc[m]
        lines.append(f"{sc if k == 0 else ''} & {lab} & {r.intensity_mean:.2f} & {r.intensity_q10:.2f} & "
                     f"{r.intensity_median:.2f} & {r.intensity_q90:.2f} & {r.frac_intensity_0:.3f} & {r.frac_intensity_1:.3f} \\\\")
    a = g.loc["ALS"]
    lines.append(f" & ALS (WLS chosen) & {a.intensity_mean:.3f} & & & & & \\\\")
    if sc != "S4":
        lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
with open(os.path.join(OUT, "tab_C123_intensity_dist.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("intensity distribution table written")
