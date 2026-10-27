"""
wls_lib.py
==========
Core library for the empirical study
    "When Does Variance Weighting Help Prediction? Feasible GLS, Adapted
     Intensity Rules, Pre-Testing and Cross-Validated Weighting under
     Mean-Function Misspecification" (IEEE Access).

Contents
--------
1. Dataset loading / preprocessing (Auto MPG, Insurance, Hitters, Engel, Abalone)
2. Iterative Weighted Least Squares (Iterative WLS) estimator
3. Auxiliary variance models: Linear / Poly2 / Poly3 / Random Forest
4. Heteroscedasticity diagnostics (Breusch-Pagan, White, decile variance ratio)
5. Prediction-interval metrics (PICP, PINAW, conditional coverage)

All experiments are fully reproducible: every stochastic component is seeded
with a fixed RANDOM_SEED defined below.

Python : 3.10+  (numpy, pandas, scikit-learn, statsmodels, scipy)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures

from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.30
N_ITER: int = 5           # default number of alternating mean-variance updates
DELTA: float = 1e-6       # numerical floor preventing log(0) and 1/0
RF_TREES: int = 200       # trees in the random-forest variance model

from _paths import stage_root
HERE = str(stage_root() / "code")
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "figures"))
for _d in (RESULTS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Datasets
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """A regression benchmark with its modelling metadata."""
    name: str
    domain: str
    X: np.ndarray                  # (n, d) design matrix, raw units
    y: np.ndarray                  # (n,) response, raw units
    feature_names: List[str]
    target_name: str
    source: str

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def summary(self) -> Dict:
        return dict(
            dataset=self.name,
            domain=self.domain,
            n=self.n,
            d=self.d,
            target=self.target_name,
            features=", ".join(self.feature_names),
            source=self.source,
            y_mean=float(np.mean(self.y)),
            y_std=float(np.std(self.y, ddof=1)),
            y_min=float(np.min(self.y)),
            y_max=float(np.max(self.y)),
        )


def load_all_datasets(data_dir: str = DATA_DIR) -> Dict[str, Dataset]:
    """Load the five benchmark datasets in their original measurement units."""

    # --- Auto MPG --------------------------------------------------------- #
    auto = pd.read_csv(os.path.join(data_dir, "Auto.csv"))
    auto["horsepower"] = pd.to_numeric(auto["horsepower"], errors="coerce")
    auto = auto.dropna(subset=["mpg", "displacement", "weight"])
    ds_auto = Dataset(
        name="Auto MPG",
        domain="Energy efficiency",
        X=auto[["displacement", "weight"]].to_numpy(float),
        y=auto["mpg"].to_numpy(float),
        feature_names=["displacement", "weight"],
        target_name="mpg (fuel economy)",
        source="ISLR / UCI Auto-Mpg",
    )

    # --- Insurance -------------------------------------------------------- #
    ins = pd.read_csv(os.path.join(data_dir, "insurance.csv"))
    ds_ins = Dataset(
        name="Insurance",
        domain="Medical insurance",
        X=ins[["age", "bmi"]].to_numpy(float),
        y=ins["charges"].to_numpy(float),
        feature_names=["age", "bmi"],
        target_name="charges (USD)",
        source="Medical Cost Personal Dataset",
    )

    # --- Hitters ---------------------------------------------------------- #
    hit = pd.read_csv(os.path.join(data_dir, "Hitters.csv"))
    hit["Salary"] = pd.to_numeric(hit["Salary"], errors="coerce")
    hit = hit.dropna(subset=["Salary", "Hits", "HmRun"])
    ds_hit = Dataset(
        name="Hitters",
        domain="Sports analytics",
        X=hit[["Hits", "HmRun"]].to_numpy(float),
        y=hit["Salary"].to_numpy(float),
        feature_names=["Hits", "HmRun"],
        target_name="Salary (1000 USD)",
        source="ISLR Hitters",
    )

    # --- Engel ------------------------------------------------------------ #
    eng = pd.read_csv(os.path.join(data_dir, "engel.csv"))
    ds_eng = Dataset(
        name="Engel",
        domain="Household economics",
        X=eng[["income"]].to_numpy(float),
        y=eng["foodexp"].to_numpy(float),
        feature_names=["income"],
        target_name="foodexp (Belgian francs)",
        source="quantreg::engel (Koenker & Bassett, 1982)",
    )

    # --- Abalone ---------------------------------------------------------- #
    abl = pd.read_csv(os.path.join(data_dir, "abalone.csv"))
    ds_abl = Dataset(
        name="Abalone",
        domain="Marine biology (UCI)",
        X=abl[["Length", "Diameter", "Height",
               "Whole weight", "Shucked weight",
               "Viscera weight", "Shell weight"]].to_numpy(float),
        y=abl["Rings"].to_numpy(float),
        feature_names=["Length", "Diameter", "Height",
                       "Whole weight", "Shucked weight",
                       "Viscera weight", "Shell weight"],
        target_name="Rings (age proxy)",
        source="UCI ML Repository, Nash et al. (1994)",
    )

    return {
        "Auto MPG": ds_auto,
        "Insurance": ds_ins,
        "Hitters": ds_hit,
        "Engel": ds_eng,
        "Abalone": ds_abl,
    }


# --------------------------------------------------------------------------- #
# 2. Auxiliary variance models  log sigma^2(X) = f_theta(X)
# --------------------------------------------------------------------------- #
VARIANCE_MODELS: Tuple[str, ...] = ("Linear", "Poly2", "Poly3", "RandomForest")


def make_variance_model(kind: str, seed: int = RANDOM_SEED):
    """Return (estimator, optional polynomial expander) for the log-variance
    auxiliary regression."""
    if kind == "Linear":
        return LinearRegression(), None
    if kind == "Poly2":
        return LinearRegression(), PolynomialFeatures(degree=2, include_bias=False)
    if kind == "Poly3":
        return LinearRegression(), PolynomialFeatures(degree=3, include_bias=False)
    if kind == "RandomForest":
        return (
            RandomForestRegressor(
                n_estimators=RF_TREES,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=-1,
            ),
            None,
        )
    raise ValueError(f"Unknown variance model: {kind}")


def _expand(expander, X: np.ndarray) -> np.ndarray:
    return X if expander is None else expander.fit_transform(X)


# --------------------------------------------------------------------------- #
# 3. Iterative Weighted Least Squares
# --------------------------------------------------------------------------- #
@dataclass
class IterativeWLS:
    """Alternating mean-variance optimisation for heteroscedastic regression.

    Parameters
    ----------
    variance_model : {'Linear','Poly2','Poly3','RandomForest'}
        Complexity of the auxiliary log-variance regressor f_theta.
    n_iter : int
        Number of alternating updates T.  T = 0 reduces to OLS.
    delta : float
        Numerical floor for log(e^2 + delta) and for the weight denominator.
    clip : float
        Upper bound applied to exp(z_hat) to keep the weights bounded away
        from zero (prevents a single near-zero residual from dominating).
    """

    variance_model: str = "Linear"
    n_iter: int = N_ITER
    delta: float = DELTA
    clip: Optional[float] = None
    seed: int = RANDOM_SEED
    temper: float = 1.0
    """Tempering exponent gamma in w_i = (1 / sigma^2(x_i))^gamma.

    gamma = 1 recovers full feasible inverse-variance weighting;
    gamma = 0 recovers OLS.  Intermediate values guard against
    misspecification of the variance model and are selected by
    cross-validation (see ``select_temper``).
    """

    # fitted attributes ---------------------------------------------------- #
    coef_: np.ndarray = field(default=None, init=False)      # (d+1,) incl. intercept
    var_model_: object = field(default=None, init=False)
    expander_: object = field(default=None, init=False)
    history_: List[float] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ #
    def _design(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IterativeWLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = X.shape[0]
        A = self._design(X)
        w = np.ones(n)

        self.history_ = []
        self.var_model_, self.expander_ = None, None

        for t in range(self.n_iter):
            # ---- (a) closed-form weighted mean update ------------------ #
            Aw = A * np.sqrt(w)[:, None]
            yw = y * np.sqrt(w)
            beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
            resid = y - A @ beta
            self.history_.append(float(np.mean(resid ** 2)))

            if t == self.n_iter - 1:      # last iteration: no need to refit sigma
                self.coef_ = beta
                break

            # ---- (b) auxiliary regression on log squared residuals ----- #
            z = np.log(resid ** 2 + self.delta)
            model, expander = make_variance_model(self.variance_model, self.seed)
            Z = _expand(expander, X)
            model.fit(Z, z)
            z_hat = model.predict(Z)

            # ---- (c) inverse-variance weights -------------------------- #
            sigma2 = np.exp(z_hat)
            if self.clip is not None:
                sigma2 = np.clip(sigma2, None, self.clip)
            w = (1.0 / (sigma2 + self.delta)) ** self.temper
            w = w / np.mean(w)            # scale-free normalisation

            self.var_model_, self.expander_ = model, expander

        if self.coef_ is None:            # n_iter == 0 -> OLS
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            self.coef_ = beta
        elif self.n_iter > 0:
            # Align the returned variance function with the returned mean.
            # The loop deliberately skips a weight refresh after the final
            # mean solve, but predictive scores must not use a variance model
            # fitted to the penultimate residuals.
            resid = y - A @ self.coef_
            z = np.log(resid ** 2 + self.delta)
            model, expander = make_variance_model(self.variance_model, self.seed)
            Z = _expand(expander, X)
            model.fit(Z, z)
            self.var_model_, self.expander_ = model, expander
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        A = self._design(np.asarray(X, dtype=float))
        return A @ self.coef_

    def predict_sigma2(self, X: np.ndarray) -> np.ndarray:
        """Estimated conditional variance sigma^2(x) on training scaling."""
        X = np.asarray(X, dtype=float)
        if self.var_model_ is None:
            resid_var = np.mean(self.history_) if self.history_ else 1.0
            return np.full(X.shape[0], resid_var)
        Z = self.expander_.transform(X) if self.expander_ is not None else X
        return np.exp(self.var_model_.predict(Z))


# --------------------------------------------------------------------------- #
# 3b. Tempered WLS -- cross-validated selection of the tempering exponent
# --------------------------------------------------------------------------- #
TEMPER_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def select_temper(X: np.ndarray, y: np.ndarray, variance_model: str = "Linear",
                  grid: Sequence[float] = TEMPER_GRID, n_iter: int = N_ITER,
                  n_splits: int = 5, seed: int = RANDOM_SEED
                  ) -> Tuple[float, pd.DataFrame]:
    """Pick gamma in w_i = (1/sigma^2(x_i))^gamma by K-fold CV.

    Returns the selected gamma and the full CV profile.  gamma = 0 is OLS,
    so OLS is always one of the candidates; selection noise means that the
    selected fit can still have higher test risk than OLS.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for g in grid:
        errs = []
        for tr, va in kf.split(X):
            m = IterativeWLS(variance_model=variance_model, n_iter=n_iter,
                             temper=g, seed=seed).fit(X[tr], y[tr])
            errs.append(mse(y[va], m.predict(X[va])))
        rows.append(dict(gamma=g, cv_mse=float(np.mean(errs)),
                         cv_sd=float(np.std(errs, ddof=1))))
    prof = pd.DataFrame(rows)
    best = float(prof.loc[prof.cv_mse.idxmin(), "gamma"])
    return best, prof


# --------------------------------------------------------------------------- #
# 4. Heteroscedasticity diagnostics
# --------------------------------------------------------------------------- #
def heteroscedasticity_tests(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Breusch-Pagan and White tests plus a decile variance ratio.

    The variance ratio is computed by binning the training residuals into
    deciles of the first principal predictor and taking
    Var(resid | top decile) / Var(resid | bottom decile).
    """
    import statsmodels.api as sm

    A = sm.add_constant(X)
    ols = sm.OLS(y, A).fit()
    resid = ols.resid
    exog = ols.model.exog

    bp_stat, bp_p, _, _ = het_breuschpagan(resid, exog)
    wh_stat, wh_p, _, _ = het_white(resid, exog)

    # decile variance ratio on the strongest marginal predictor
    if X.shape[1] > 1:
        corr = np.corrcoef(X.T, y)[:-1, -1]       # corr(feature_j, y)
        order = np.argsort(-np.abs(corr))
    else:
        order = [0]
    driver = X[:, int(order[0])]
    q = pd.qcut(driver, 10, labels=False, duplicates="drop")
    lo = np.var(resid[q == 0], ddof=1)
    hi = np.var(resid[q == q.max()], ddof=1)
    ratio = hi / lo if lo > 0 else np.nan

    return dict(
        bp_stat=float(bp_stat),
        bp_p=float(bp_p),
        white_stat=float(wh_stat),
        white_p=float(wh_p),
        var_ratio=float(ratio),
        resid_sd=float(np.std(resid, ddof=1)),
    )


# --------------------------------------------------------------------------- #
# 5. Prediction-interval metrics
# --------------------------------------------------------------------------- #
def picp(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Prediction Interval Coverage Probability."""
    return float(np.mean((y_true >= lo) & (y_true <= hi)) * 100.0)


def pinaw(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Prediction Interval Normalised Average Width (% of response range)."""
    rng = np.max(y_true) - np.min(y_true)
    return float(np.mean(hi - lo) / rng * 100.0)


def conditional_coverage(y_true, lo, hi, sigma_hat, n_groups: int = 3) -> pd.DataFrame:
    """Coverage stratified by the estimated noise level sigma_hat(x)."""
    q = pd.qcut(sigma_hat, n_groups, labels=False, duplicates="drop")
    if not np.isfinite(q).all() or len(np.unique(q)) != n_groups:
        raise ValueError("Predicted scales do not define the requested noise strata")
    rows = []
    for g in sorted(np.unique(q)):
        m = q == g
        rows.append(
            dict(
                group=f"Q{g + 1} ({'low' if g == 0 else 'high' if g == q.max() else 'mid'} sigma)",
                n=int(m.sum()),
                coverage_pct=picp(y_true[m], lo[m], hi[m]),
                mean_width=float(np.mean((hi - lo)[m])),
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Baseline factory
# --------------------------------------------------------------------------- #
def make_baseline(name: str, seed: int = RANDOM_SEED):
    if name == "OLS":
        return LinearRegression()
    if name == "Ridge":
        return RidgeCVLite()
    if name == "Lasso":
        return LassoCVLite(random_state=seed)
    if name == "ElasticNet":
        return ElasticNetCVLite(random_state=seed)
    raise ValueError(name)


class _CVBase:
    """Thin wrapper selecting the penalty by 5-fold CV on the training fold.

    A hand-rolled grid is used (rather than sklearn's *CV classes) so that the
    grid is identical across datasets and the selected penalty is logged.
    """

    alphas: np.ndarray = np.logspace(-4, 4, 41)

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.alpha_ = None

    def fit(self, X, y):
        from sklearn.model_selection import KFold

        best, best_a = np.inf, self.alphas[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for a in self.alphas:
            errs = []
            for tr, va in kf.split(X):
                m = self._make(a)
                m.fit(X[tr], y[tr])
                errs.append(np.mean((y[va] - m.predict(X[va])) ** 2))
            e = float(np.mean(errs))
            if e < best:
                best, best_a = e, a
        self.alpha_ = float(best_a)
        self.model_ = self._make(best_a)
        self.model_.fit(X, y)
        return self

    def _make(self, alpha):            # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def coef_(self):
        return self.model_.coef_


class RidgeCVLite(_CVBase):
    def _make(self, alpha):
        return Ridge(alpha=alpha)


class LassoCVLite(_CVBase):
    alphas = np.logspace(-2, 4, 31)

    def _make(self, alpha):
        return Lasso(alpha=alpha, max_iter=50000)


class ElasticNetCVLite(_CVBase):
    alphas = np.logspace(-2, 4, 31)

    def _make(self, alpha):
        return ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=50000)


# --------------------------------------------------------------------------- #
# 7. Small utilities
# --------------------------------------------------------------------------- #
def mse(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def train_test_split_idx(n: int, test_size: float = TEST_SIZE,
                         seed: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_size))
    return idx[n_test:], idx[:n_test]


def fmt_sci(v: float) -> str:
    """Compact scientific formatting for very large MSEs in tables."""
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.4g}"
    return f"{v:,.4f}"
