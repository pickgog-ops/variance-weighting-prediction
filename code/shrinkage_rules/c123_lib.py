"""Core comparison library for the intensity-rule experiments.

The iterative AMV rules share the same linear log-variance auxiliary
regression, numerical floor delta=1e-6, T=5 mean-fit cycles (initial OLS plus
four weighted updates), unit-mean weight normalisation and the final scale
refit aligned to the returned mean, exactly as wls_lib.IterativeWLS.  Only the
map from the fitted log-variance to the weights, and how its intensity is
chosen, differs between these iterative methods. ALS instead uses an OLS pretest and a single WLS update.
"""
from __future__ import annotations

import sys
import types
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Use the installed diagnostics dependency; no stub modules are registered.
import statsmodels.stats.diagnostic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import wls_lib as W  # noqa: E402

DELTA = 1e-6
T_CYCLES = 5
LOG_CORRECTION = 1.2704
S2_FLOOR = 1e-12
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
ALS_ALPHA = 0.10


# --------------------------------------------------------------------------- #
# Weight maps.  Each receives the training design X (n,d), the auxiliary
# log-squared-residual response z, the fitted log-variance f_hat and the
# fitted auxiliary model, and returns (weights, intensity).
# --------------------------------------------------------------------------- #
def aux_stats(X, z, f_hat):
    """Residual variance of the auxiliary regression and hat-matrix diagonal."""
    n, d = X.shape
    A = np.column_stack([np.ones(n), X])
    tau2 = float(np.sum((z - f_hat) ** 2) / (n - d - 1))
    # leverage h_ii of the auxiliary design (with intercept)
    Qm, _ = np.linalg.qr(A)
    h = np.sum(Qm ** 2, axis=1)
    return tau2, h


def map_power(gamma):
    def f(X, z, f_hat):
        return (np.exp(f_hat) + DELTA) ** (-gamma), gamma
    return f


def map_eb_power(X, z, f_hat):
    """James-Stein / empirical-Bayes linear shrinkage of fitted log-variances
    towards their mean.  The noise contribution to the dispersion of the d
    fitted slopes' predictions is d*tau2 (trace of the centred hat matrix);
    kappa = max(0, 1 - d*tau2/SS).  Shrinking log-variances by kappa equals a
    power map with exponent kappa (up to a common multiplier)."""
    n, d = X.shape
    tau2, _ = aux_stats(X, z, f_hat)
    ss = float(np.sum((f_hat - f_hat.mean()) ** 2))
    kappa = 0.0 if ss <= 0 else max(0.0, 1.0 - d * tau2 / ss)
    return (np.exp(f_hat) + DELTA) ** (-kappa), kappa


def map_linear(rho):
    def f(X, z, f_hat):
        s = np.exp(f_hat)
        nu = float(np.mean(s))
        return 1.0 / ((1.0 - rho) * s + rho * nu + DELTA), rho
    return f


def map_lw_linear(X, z, f_hat):
    """Ledoit-Wolf-type analytic intensity for diagonal linear shrinkage of the
    fitted variances towards their mean nu: rho = sum Var(s_i) / sum (s_i-nu)^2,
    with the delta-method estimate Var(s_i) ~= s_i^2 * tau2 * h_ii, truncated to
    [0,1].  Weights are bounded above by 1/(rho*nu)."""
    tau2, h = aux_stats(X, z, f_hat)
    s = np.exp(f_hat)
    nu = float(np.mean(s))
    den = float(np.sum((s - nu) ** 2))
    num = float(np.sum(s ** 2 * tau2 * h))
    rho = 1.0 if den <= 0 else min(1.0, max(0.0, num / den))
    return 1.0 / ((1.0 - rho) * s + rho * nu + DELTA), rho


# --------------------------------------------------------------------------- #
# Generalised alternating mean-variance fit (mirrors wls_lib.IterativeWLS)
# --------------------------------------------------------------------------- #
class AMV:
    def __init__(self, weight_map, n_iter=T_CYCLES):
        self.weight_map = weight_map
        self.n_iter = n_iter

    @staticmethod
    def _design(X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n = X.shape[0]
        A = self._design(X)
        w = np.ones(n)
        self.intensities_ = []
        self.coef_ = None
        for t in range(self.n_iter):
            Aw = A * np.sqrt(w)[:, None]
            beta, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
            resid = y - A @ beta
            if t == self.n_iter - 1:
                self.coef_ = beta
                break
            z = np.log(resid ** 2 + DELTA)
            m = LinearRegression().fit(X, z)
            f_hat = m.predict(X)
            w, inten = self.weight_map(X, z, f_hat)
            w = w / np.mean(w)
            self.intensities_.append(float(inten))
        resid = y - A @ self.coef_
        self.scale_ = LinearRegression().fit(X, np.log(resid ** 2 + DELTA))
        return self

    @property
    def intensity_(self):
        return self.intensities_[-1] if self.intensities_ else np.nan

    def predict(self, X):
        return self._design(np.asarray(X, float)) @ self.coef_

    def predict_sigma2(self, X):
        return np.exp(self.scale_.predict(np.asarray(X, float)))


class ALS:
    """Adaptation of Romano & Wolf (2017) adaptive least squares: OLS, then auxiliary
    regression of log-squared OLS residuals on the regressors (exponential
    skedastic form); n*R^2 test against chi2(d) at level 0.10; if it rejects,
    one-step WLS with weights 1/v_hat, otherwise OLS."""

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n, d = X.shape
        A = np.column_stack([np.ones(n), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        z = np.log(resid ** 2 + DELTA)
        m = LinearRegression().fit(X, z)
        f_hat = m.predict(X)
        r2 = 1.0 - np.sum((z - f_hat) ** 2) / np.sum((z - z.mean()) ** 2)
        self.stat_ = float(n * r2)
        self.reject_ = bool(self.stat_ > stats.chi2.ppf(1 - ALS_ALPHA, d))
        if self.reject_:
            w = 1.0 / (np.exp(f_hat) + DELTA)
            w = w / np.mean(w)
            Aw = A * np.sqrt(w)[:, None]
            beta, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
        self.coef_ = beta
        resid = y - A @ beta
        self.scale_ = LinearRegression().fit(X, np.log(resid ** 2 + DELTA))
        return self

    @property
    def intensity_(self):
        return 1.0 if self.reject_ else 0.0

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), np.asarray(X, float)]) @ self.coef_

    def predict_sigma2(self, X):
        return np.exp(self.scale_.predict(np.asarray(X, float)))


def cv_select(X, y, map_factory, grid=GRID, seed=42, n_splits=5):
    """Inner K-fold selection with the same KFold(shuffle, random_state=seed)
    construction as wls_lib.select_temper; ties resolved by the first minimum."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(kf.split(X))
    cv = []
    for g in grid:
        errs = []
        for tr, va in folds:
            m = AMV(map_factory(g)).fit(X[tr], y[tr])
            errs.append(W.mse(y[va], m.predict(X[va])))
        cv.append(float(np.mean(errs)))
    best = float(grid[int(np.argmin(cv))])
    return best, cv


def gaussian_scores(y, mu, raw_s2):
    """MSE/CRPS/NLL with the corrected learned scale (as in the reference groups C/D)."""
    s2 = np.maximum(raw_s2 * np.exp(LOG_CORRECTION), S2_FLOOR)
    sd = np.sqrt(s2)
    zz = (y - mu) / sd
    se = (y - mu) ** 2
    nll = 0.5 * np.log(2 * np.pi * s2) + se / (2 * s2)
    crps = sd * (zz * (2 * stats.norm.cdf(zz) - 1) + 2 * stats.norm.pdf(zz) - 1 / np.sqrt(np.pi))
    return float(se.mean()), float(crps.mean()), float(nll.mean())
