# When does variance weighting help prediction? Code and data

Code, data and saved results for the main experiments of the paper

> X. Ding, F. Yang, B. Liu, T. Tu and T. Feng, "When Does Variance Weighting Help Prediction? Feasible GLS, Adapted Intensity Rules, Pre-Testing and Cross-Validated Weighting under Mean-Function Misspecification," *IEEE Access*, 2026 (manuscript Access-xxxx-xxxxx; ).

The repository covers the Monte Carlo study (Table 2), the weighting-intensity rules (Table 3 and Appendix E), the benchmark comparison (Tables 4-6) and the correlated-error GLS study (Appendix J). The supplementary material of the article contains the complete archive with all further analyses.

## Quick check (about one minute)

Python 3.11 or later. In this folder:

```text
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (macOS / Linux)
python -m pip install -r requirements.txt
python run_checks.py
```

`run_checks.py` recomputes the paired tests, the table values and the LaTeX tables from the saved fold- and replication-level results and compares every output with the saved file. It ends with `All checks passed.` No model is refitted and no file of the repository is changed; outputs go to a new folder `runs/check_<date-time>/`. For the library versions of the reference run use `requirements-tested.txt`.

## Contents

| Path | Content | Paper |
|---|---|---|
| `data/` | Five benchmark data sets (`DATA_SOURCES.md`) | Table 1 |
| `code/main/`, `results/main/` | OLS, FGLS (`WLS-ML` in the files), TWLS and standardised penalised baselines on 10 x 5-fold CV; scoring rules; prediction intervals; Monte Carlo scenarios S1-S4 | Tables 2, 4, 5 and 6 |
| `code/shrinkage_rules/`, `results/shrinkage_rules/` | Seven weighting-intensity rules in simulation and on the benchmarks | Table 3, Appendix E |
| `reference/` | Per-fold losses with the learned variance scale (OLS+Scale, FGLS, TWLS) and per-replication simulation results used as inputs and for implementation checks (see "The `reference/` files" below) | Tables 2 and 4-6 |
| `correlated_gls_study/` | Ledoit-Wolf linear, OAS, analytical nonlinear and QIS covariance shrinkage in correlated-error GLS (own README) | Appendix J |
| `reproduce.py` | Runs a stage in a separate work folder | |
| `run_checks.py` | Quick check (above) | |

The values of the main-text Tables 2, 4, 5 and 6 are collected in `results/main/main_text_values.csv`; the paired tests behind their significance marks are in `results/main/tests_benchmarks_holm.csv` (Nadeau-Bengio corrected t tests on the 50 paired fold losses, Holm adjustment within each metric).

## The `reference/` files

`reference/realdata_reference_metrics.csv` holds the per-fold losses of OLS with learned scale (group B) and of FGLS and CV-TWLS with their own learned scales (groups C and D), on the 50 outer folds (`cv50`) and on the 20 interval splits (`holdout20`). `reference/simulation_reference_replications.csv` holds the 800 regenerated replication records of scenarios S1-S4 (Section V-C and Appendix C of the article). The scripts that wrote these files belong to the complete archive in the supplementary material. They can be checked with this repository: `python reproduce.py run --stage shrinkage_rules --action full` refits OLS, FGLS and CV-TWLS on the same 50 folds, compares the refitted losses with groups B, C and D (`results/check_C123_realdata_vs_reference.csv`), and its simulation part refits the S1-S4 replications at n = 500.

## Running a stage

```text
python reproduce.py prepare --stage all --work-dir runs/replay --saved
python reproduce.py run --stage main --action analyze --work-dir runs/replay
python reproduce.py run --stage shrinkage_rules --action analyze --work-dir runs/replay
python reproduce.py run --stage shrinkage_rules --action tables --work-dir runs/replay
```

`prepare` copies code, data, reference files and (with `--saved`) the saved results into `runs/replay/<stage>/`; all outputs are written there.

To refit all models, prepare a new folder without `--saved` and use `--action full`:

```text
python reproduce.py prepare --stage all --work-dir runs/full
python reproduce.py run --stage main --action full --work-dir runs/full
python reproduce.py run --stage shrinkage_rules --action full --work-dir runs/full
```

This is much slower than the replay. Seeds and folds are fixed in the code (outer folds `RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)`), so the results should agree with the saved files up to floating-point differences between platforms. `run_experiments.py` also writes the 70/30-split point comparison of the original submission (`tab3_main.csv`, `raw_main_splits.csv`), which is not reported in the final article; its interval study (`tab6_intervals.csv`) is used for Table 6.

## Licence and data

The code is provided for research use and for checking the results of the paper. The data sets keep the terms of their original sources (`data/DATA_SOURCES.md`); third-party estimator code in `correlated_gls_study/` keeps its own licences (`correlated_gls_study/THIRD_PARTY_NOTICE.md`).
