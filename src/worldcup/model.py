"""Train, calibrate, and evaluate the World Cup match model.

Design choices that follow the repo's CLAUDE.md:

* **Temporal split, never random.** Train on all competitive internationals
  before the target World Cup; test on the tournament itself. The 2018 and
  2022 tournaments are wholly out-of-sample future events.
* **Orientation invariance.** World Cup venues are neutral, and the source
  match orientation (which team is listed "home") is an arbitrary artefact.
  The training set is mirrored (teams swapped, features negated, H<->A) so the
  model cannot exploit listing order; a signed ``home_field`` term carries the
  real host advantage.
* **Baselines to beat.** Class-prior (the sanity floor) and an Elo-only model
  (the strong baseline; bookmaker odds are unavailable for historical World
  Cups in the free sources, and Elo is the standard international-football
  analogue). The full model must beat both on log-loss and Brier.
* **Metrics order.** Multi-class log-loss (primary), Brier (secondary),
  accuracy, reported via ``src.analysis.evaluation``.

Calibration: a temporal tail of the training set is held out and sigmoid
(Platt) calibration is fit there (never on the test tournament). Sigmoid is
chosen over isotonic because it yields smooth probabilities that never collapse
a class to a hard 0 on the small per-tournament test set.
"""

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
    # Sigmoid (Platt) calibration: fits a smooth logistic map per class, so
    # probabilities stay strictly between 0 and 1. Preferred here over isotonic,
    # which produces step functions that can collapse a class (e.g. draw) to a
    # hard 0 on the small per-tournament test set.
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
    """Train on pre-tournament data, evaluate on the World Cup.

    Returns a dict with a ``results`` DataFrame (one row per model: full,
    Elo-only baseline, class-prior baseline), the fitted full model, the test
    frame with predicted probabilities, and the data bundle.
    """
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

    # Columns are named team1/team2, not home/away: World Cup venues are
    # neutral, so "team1" is just the orientation stored in the source row, not
    # a team with home advantage. Real host advantage is the ``home_field``
    # feature, and the model is symmetrised so orientation carries no signal.
    test = test.copy()
    test[["p_team1", "p_draw", "p_team2"]] = p_full
    return {"results": results, "model": full, "test": test, "data": data,
            "n_train": len(train)}
