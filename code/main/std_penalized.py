"""Standardised ridge, lasso and elastic net on the 50 outer folds of the main
comparison (RepeatedKFold(5, 10, random_state=42)).

Predictors and response are standardised with training-fold moments and the
penalty is chosen by inner five-fold CV. Predictive distributions are
homoscedastic Gaussian with the training residual variance, as for OLS.
These are the Ridge, Lasso and ENet columns of the main-text MSE and CRPS tables.

Output: results/raw_std_penalized.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, RepeatedKFold

from wls_lib import load_all_datasets, RESULTS_DIR

GRIDS = {"Ridge": np.logspace(-4, 4, 41), "Lasso": np.logspace(-4, 2, 31),
         "ElasticNet": np.logspace(-4, 2, 31)}


class StdPenalized:
    def __init__(self, kind, random_state=0):
        self.kind, self.random_state = kind, random_state

    def _make(self, a):
        if self.kind == "Ridge":
            return Ridge(alpha=a)
        if self.kind == "Lasso":
            return Lasso(alpha=a, max_iter=50000)
        return ElasticNet(alpha=a, l1_ratio=0.5, max_iter=50000)

    def _fit_one(self, X, y, a):
        mx, sx = X.mean(0), X.std(0)
        sx = np.where(sx > 0, sx, 1.0)
        my, sy = y.mean(), y.std() or 1.0
        m = self._make(a).fit((X - mx) / sx, (y - my) / sy)
        return lambda Xn: my + sy * m.predict((Xn - mx) / sx)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        folds = list(KFold(n_splits=5, shuffle=True, random_state=self.random_state).split(X))
        best, best_a = np.inf, None
        for a in GRIDS[self.kind]:
            e = np.mean([np.mean((y[va] - self._fit_one(X[tr], y[tr], a)(X[va])) ** 2) for tr, va in folds])
            if e < best:
                best, best_a = e, a
        self.alpha_ = float(best_a)
        self._pred = self._fit_one(X, y, best_a)
        return self

    def predict(self, X):
        return self._pred(np.asarray(X, float))


def homo_scores(ytr, fit_tr, yte, mu):
    s2 = np.var(ytr - fit_tr, ddof=1)
    sd = np.sqrt(s2)
    z = (yte - mu) / sd
    nll = float(np.mean(0.5 * np.log(2 * np.pi * s2) + (yte - mu) ** 2 / (2 * s2)))
    crps = float(np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))
    return float(np.mean((yte - mu) ** 2)), crps, nll


rows = []
for name, v in load_all_datasets().items():
    for k, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=10, random_state=42).split(v.X)):
        Xtr, ytr, Xte, yte = v.X[tr], v.y[tr], v.X[te], v.y[te]
        for m in ("Ridge", "Lasso", "ElasticNet"):
            est = StdPenalized(m, random_state=k).fit(Xtr, ytr)
            mse, crps, nll = homo_scores(ytr, est.predict(Xtr), yte, est.predict(Xte))
            rows.append(dict(dataset=name, split_id=k, method=m + "-std", mse=mse, crps=crps, nll=nll, alpha=est.alpha_))
    print("done", name, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "raw_std_penalized.csv"), index=False)
