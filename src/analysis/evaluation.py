"""Probabilistic forecast evaluation for 1X2 football predictions.

References
----------
Wheatcroft, E. (2021), "Evaluating probabilistic forecasts of football
matches: the case against the ranked probability score", JQAS 17(4).

Why these metrics, in this order
--------------------------------
The repository's CLAUDE.md prescribes multi-class log-loss as the
primary metric and Brier as secondary. Wheatcroft argues that the
Ignorance score (which is exactly multi-class log-loss with log base 2)
dominates RPS in football contexts. This module reports log-loss,
Brier, RPS, and accuracy together so reports can compare against the
literature without re-implementing each one.

Baselines
---------
Two baselines per the CLAUDE.md spec:

1. Home-win prior  -- predict the empirical class prior every match.
2. Bookmaker-implied -- decimal odds -> implied probabilities with the
   overround removed proportionally (the simplest, most-cited
   normalisation; see Strumbelj 2014 for alternatives).

The ``evaluate`` helper compares any candidate probability matrix
against these baselines on the same rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Class labels ─────────────────────────────────────────────────────


CLASS_ORDER = ("home_win", "draw", "away_win")
RESULT_TO_INDEX = {"H": 0, "D": 1, "A": 2}


def result_to_y(results: pd.Series | np.ndarray) -> np.ndarray:
    """Map a Series of 'H'/'D'/'A' labels to a numpy int array {0,1,2}."""
    arr = np.asarray(results)
    return np.vectorize(RESULT_TO_INDEX.__getitem__)(arr)


# ── Scoring rules ────────────────────────────────────────────────────


def log_loss(proba: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Multi-class log-loss in nats (natural log).

    The CLAUDE.md spec names this as the primary metric.
    """
    proba = np.clip(np.asarray(proba, dtype=float), eps, 1.0)
    return float(-np.mean(np.log(proba[np.arange(len(y)), y])))


def brier_multiclass(proba: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score, summed over the three class indicators."""
    proba = np.asarray(proba, dtype=float)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(((proba - one_hot) ** 2).sum(axis=1)))


def ranked_probability_score(proba: np.ndarray, y: np.ndarray) -> float:
    """Three-class RPS using the ordinal H < D < A convention.

    Reported alongside log-loss for comparability with the literature.
    Wheatcroft (2021) argues log-loss should be preferred.
    """
    proba = np.asarray(proba, dtype=float)
    cum_pred = np.cumsum(proba, axis=1)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y)), y] = 1.0
    cum_true = np.cumsum(one_hot, axis=1)
    # Only first K-1 partial sums contribute (the last column is always 1).
    return float(np.mean(((cum_pred[:, :-1] - cum_true[:, :-1]) ** 2).sum(axis=1)))


def accuracy(proba: np.ndarray, y: np.ndarray) -> float:
    """Top-1 classification accuracy (secondary; see CLAUDE.md ordering)."""
    return float((np.argmax(np.asarray(proba), axis=1) == y).mean())


# ── Bookmaker baseline ───────────────────────────────────────────────


def odds_to_proba(
    odds_home: np.ndarray,
    odds_draw: np.ndarray,
    odds_away: np.ndarray,
) -> np.ndarray:
    """Convert decimal odds to probabilities, removing the overround.

    Uses proportional normalisation: divide each 1/odds by the row sum.
    Returns an (n, 3) array with columns in CLASS_ORDER. Rows containing
    any non-positive or NaN odds are returned as NaN so the caller can
    decide how to handle them.
    """
    arr = np.column_stack([odds_home, odds_draw, odds_away]).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / arr
    bad = (arr <= 0) | ~np.isfinite(arr)
    raw[bad.any(axis=1)] = np.nan
    row_sums = raw.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = raw / row_sums
    return normalised


# ── Reliability curves ───────────────────────────────────────────────


def reliability_curve(
    proba: np.ndarray,
    y: np.ndarray,
    class_index: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability data for a single class.

    Returns columns: bin_lower, bin_upper, n, mean_predicted, frac_positive.
    """
    p = np.asarray(proba)[:, class_index]
    is_pos = (y == class_index).astype(int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bin_id == b
        if mask.sum() == 0:
            continue
        out.append({
            "bin_lower": edges[b],
            "bin_upper": edges[b + 1],
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "frac_positive": float(is_pos[mask].mean()),
        })
    return pd.DataFrame(out)


# ── End-to-end evaluation ────────────────────────────────────────────


@dataclass
class EvalResult:
    """Container for one model's metrics on one evaluation slice."""

    model_name: str
    split: str
    n: int
    metrics: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, float | str | int]:
        row: dict[str, float | str | int] = {
            "model": self.model_name,
            "split": self.split,
            "n": self.n,
        }
        row.update(self.metrics)
        return row


def evaluate(
    name: str,
    split: str,
    proba: np.ndarray,
    y: np.ndarray,
) -> EvalResult:
    """Compute log-loss, Brier, RPS, accuracy in CLAUDE.md order."""
    return EvalResult(
        model_name=name,
        split=split,
        n=int(len(y)),
        metrics={
            "log_loss": log_loss(proba, y),
            "brier": brier_multiclass(proba, y),
            "rps": ranked_probability_score(proba, y),
            "accuracy": accuracy(proba, y),
        },
    )


def class_prior_proba(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    """Constant-prior baseline: predict the empirical training class mix."""
    counts = np.bincount(y_train, minlength=3).astype(float)
    prior = counts / counts.sum()
    return np.tile(prior, (n_rows, 1))


def home_win_baseline(n_rows: int) -> np.ndarray:
    """Trivial 'home wins every match' baseline (probability 1.0 on H).

    Reported to anchor log-loss against a strawman; expect it to lose
    badly because it gives zero probability to draws and away wins.
    """
    base = np.full((n_rows, 3), 1e-3)
    base[:, 0] = 1.0 - 2e-3
    return base
