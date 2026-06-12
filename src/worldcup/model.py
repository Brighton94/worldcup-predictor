"""Train, calibrate, and evaluate the World Cup match model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # works whether run as a package or with repo root on sys.path
    from src.analysis import evaluation as ev
except ImportError:  # pragma: no cover
    from ..analysis import evaluation as ev

from . import config as C
from .features import FEATURES, build_dataset

ELO_ONLY = ["d_elo", "home_field"]

# H<->A swap under mirroring; D unchanged
_FLIP_Y = {0: 2, 1: 1, 2: 0}


def symmetrize(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Append the mirror of every row so the model is orientation-invariant."""
    mirror = df.copy()
    for f in feats:
        mirror[f] = -mirror[f]
    mirror["y"] = mirror["y"].map(_FLIP_Y)
    return pd.concat([df, mirror], ignore_index=True)


def _fit(train: pd.DataFrame, feats: list[str], calibrate: bool):
    """Fit a multinomial LR (optionally calibrated on a temporal tail)."""
    train = train.sort_values("date").reset_index(drop=True)
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=C.__dict__.get("RANDOM_STATE", 7)),
    )
    if not calibrate or len(train) < 400:
        sym = symmetrize(train, feats)
        base.fit(sym[feats], sym["y"])
        return base

    cut = int(len(train) * 0.85)
    fit_part, cal_part = train.iloc[:cut], train.iloc[cut:]
    sym = symmetrize(fit_part, feats)
    base.fit(sym[feats], sym["y"])
    # Sigmoid (Platt) calibration: keeps probabilities in (0,1) and avoids isotonic collapsing the draw class on small test sets.
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal_sym = symmetrize(cal_part, feats)
    cal.fit(cal_sym[feats], cal_sym["y"])
    return cal


def _metrics(name: str, proba: np.ndarray, y: np.ndarray) -> dict:
    return {
        "model": name,
        "log_loss": ev.log_loss(proba, y),
        "brier": ev.brier_multiclass(proba, y),
        "accuracy": ev.accuracy(proba, y),
    }


def train_eval(wc_year: int, calibrate: bool = True) -> dict:
    """Train on pre-tournament data, evaluate on the World Cup."""
    data = build_dataset(wc_year)
    train, test = data["train"], data["test"]
    yte = test["y"].to_numpy()

    full = _fit(train, FEATURES, calibrate)
    elo = _fit(train, ELO_ONLY, calibrate)

    p_full = full.predict_proba(test[FEATURES])
    p_elo = elo.predict_proba(test[ELO_ONLY])
    p_prior = ev.class_prior_proba(train["y"].to_numpy(), len(test))

    results = pd.DataFrame([
        _metrics("full (Elo + FIFA strength)", p_full, yte),
        _metrics("baseline: Elo-only", p_elo, yte),
        _metrics("baseline: class-prior", p_prior, yte),
    ])

    # team1/team2 are orientation only (neutral venues); host edge is the home_field feature.
    test = test.copy()
    test[["p_team1", "p_draw", "p_team2"]] = p_full
    return {"results": results, "model": full, "test": test, "data": data,
            "n_train": len(train)}
