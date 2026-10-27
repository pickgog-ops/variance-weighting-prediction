# Correlated-error GLS study (Appendix J)

Code, replication-level records and outputs for Section VI-F and Appendix J (Tables S42-S51, Fig. S9). The study compares the Ledoit-Wolf linear, OAS, analytical nonlinear (LW-NL) and QIS covariance estimators with sample, diagonal and cross-validated covariance estimates in one-step feasible GLS.

## Design

- p units observed in T independent periods: y_t = Z_t beta + m_t + eps_t, Z_t = [1, x_t], x_tj = mu_j + u_tj with u_tj ~ N(0, 1), mu_j equally spaced on [-1, 1], beta = (1, 1)'.
- eps_t ~ N_p(0, Sigma), Sigma = D^(1/2) R D^(1/2), R_ij = rho^|i-j|; unit variances log-equally spaced with max/min ratio v and mean 1, increasing with mu_j.
- Mean correctly specified (m_t = 0) or misspecified (m_tj = a(x_tj^2 - 1), a = 0.25); the fitted model always has an intercept and a linear term.
- 96 configurations: p in {10, 30, 60}, T in {20, 40, 100, 400}, rho in {0.3, 0.8}, v in {1, 20}, two mean specifications; 200 paired replications each and 100 test periods per replication. Sensitivity: the 16 configurations with p = 30 and misspecified mean repeated with a = 0.5.
- Methods: OLS, oracle GLS (true Sigma), sample GLS (T > p only), diagonal FGLS, LW-linear, OAS, LW-NL, QIS, and CV-linear / CV-power shrinkage chosen by inner five-fold CV on the training periods. All covariance estimators use the same pooled-OLS residual matrix.
- Inference: paired t tests on the 200 replications with Holm adjustment within families F1 (test MSE vs OLS), F2 (coefficient error vs OLS), F3 (original estimators vs CV rules), F4 (sensitivity, a = 0.5) and F5 (variance arrangement check).

## Code

| File | Role |
|---|---|
| `design.py` | Covariance design and exact population targets beta_0 and beta_Sigma |
| `lib.py` | Data generation, estimators, cross-validation and one paired replication |
| `official_cov1para.py`, `official_qis.py`, `official_qis_eigh.py`, `nonlinshrink_port.py` | Third-party estimator code (`THIRD_PARTY_NOTICE.md`) |
| `validate.py` | Estimator and target checks (`validation_output.txt`) |
| `run_all.py` | Simulation runs: `main` (96 configurations) or `sens` (a = 0.5) |
| `run_posthoc_alignment.py` | Variance-arrangement check (family F5) |
| `analyze.py`, `analyze_extra.py`, `analyze_posthoc.py` | Paired tests, Holm families F1-F5 and summaries |
| `feasible_target.py` | Probability limit of feasible GLS under the misspecified mean |
| `make_tables.py`, `make_fig.py` | Tables S42-S51 and Fig. S9 |

## Replaying the analysis

The replication-level records `replication_level_main.csv.gz`, `replication_level_sens.csv.gz` and `replication_level_posthoc_alignment.csv.gz` are included. In a copy of this folder run

```text
python validate.py
python analyze.py
python analyze_extra.py
python analyze_posthoc.py
python feasible_target.py
python make_tables.py
python make_fig.py
```

This reproduces every CSV file and all ten tables of the folder (`run_checks.py` in the repository root does this in a separate folder).

## Full recomputation

```text
python run_all.py main
python run_all.py sens
python run_posthoc_alignment.py
```

These write `results_main.pkl`, `results_sens.pkl` and `results_posthoc_alignment.pkl`, which the analysis scripts then use instead of the records. The main run takes about 20 minutes with two processes. Seeds are `20261012 + 1000*config_id + r` for replication r; the sensitivity runs use configuration IDs offset by 1000, the variance-arrangement runs IDs 2000-2015 with permutation seed 424242.
