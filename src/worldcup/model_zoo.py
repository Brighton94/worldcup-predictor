"""Candidate 1X2 models and a temporally-correct World Cup backtest harness.

The default model (a calibrated multinomial logistic regression) lives in
``model.py``. This module wraps several alternatives - tree ensembles, kNN,
an RBF SVM and XGBoost - behind the same calibrated fit so they can be
compared head-to-head on held-out World Cups. XGBoost is optional; if it is
not installed it is simply omitted from the line-up.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from .features import FEATURES, build_dataset
from .model import ELO_ONLY, symmetrize

try:  # works whether run as a package or with repo root on sys.path
    from src.analysis import evaluation as ev
except ImportError:  # pragma: no cover
    from ..analysis import evaluation as ev

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

RANDOM_STATE = C.__dict__.get("RANDOM_STATE", 7)


def model_specs() -> dict:
    """Name -> unfitted, probability-capable estimator (kept deliberately tame)."""
    specs = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=20,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        ),
        "knn": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=75, weights="distance"),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(), SVC(C=1.0, probability=True, random_state=RANDOM_STATE),
        ),
    }
    if _HAS_XGB:
        specs["xgboost"] = XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300,
            max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=1.0, tree_method="hist",
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
    return specs


def fit_model(estimator, train: pd.DataFrame, feats: list[str], calibrate: bool = True):
    """Fit any estimator with the same symmetrized, temporally-calibrated recipe as model.py."""
    train = train.sort_values("date").reset_index(drop=True)
    if not calibrate or len(train) < 400:
        sym = symmetrize(train, feats)
        estimator.fit(sym[feats], sym["y"])
        return estimator

    cut = int(len(train) * 0.85)
    fit_part, cal_part = train.iloc[:cut], train.iloc[cut:]
    sym = symmetrize(fit_part, feats)
    estimator.fit(sym[feats], sym["y"])
    # Sigmoid (Platt) calibration on a held-out temporal tail; sklearn >=1.6 path first.
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
    except ImportError:  # pragma: no cover
        cal = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
    cs = symmetrize(cal_part, feats)
    cal.fit(cs[feats], cs["y"])
    return cal


def backtest(wc_years: tuple[int, ...] = (2018, 2022), calibrate: bool = True) -> pd.DataFrame:
    """Backtest every candidate plus the class-prior and Elo-only baselines per World Cup."""
    rows = []
    for yr in wc_years:
        data = build_dataset(yr)
        train, test = data["train"], data["test"]
        y = test["y"].to_numpy()
        split = f"WC{yr}"

        prior = ev.class_prior_proba(train["y"].to_numpy(), len(test))
        rows.append(ev.evaluate("baseline:class-prior", split, prior, y).as_row())
        elo = fit_model(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            train, ELO_ONLY, calibrate)
        rows.append(ev.evaluate("baseline:elo-only", split, elo.predict_proba(test[ELO_ONLY]), y).as_row())

        for name, est in model_specs().items():
            m = fit_model(clone(est), train, FEATURES, calibrate)
            rows.append(ev.evaluate(name, split, m.predict_proba(test[FEATURES]), y).as_row())
    return pd.DataFrame(rows)


def fit_for_2026(name: str, calibrate: bool = True):
    """Fit one named candidate on all data up to the 2022 World Cup, ready for 2026 forecasting."""
    train = build_dataset(2022)["train"]
    return fit_model(clone(model_specs()[name]), train, FEATURES, calibrate)
