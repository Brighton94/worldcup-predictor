"""World Football Elo, computed forward in time over all international matches.

This is the project's strong baseline (the international-football analogue of
bookmaker-implied probabilities, which are unavailable for historical World
Cups in the free sources used here).

Method (the widely used eloratings.net formulation)
---------------------------------------------------
* Expected score  We = 1 / (10 ** (-dr / 400) + 1), where ``dr`` is the rating
  difference plus a home-advantage term (100 Elo) applied only at non-neutral
  venues.
* Update          R' = R + K * G * (W - We), with W in {1, 0.5, 0}.
* Goal multiplier G = 1 (margin <= 1), 1.5 (margin == 2), (11 + margin) / 8
  (margin >= 3).
* Importance K by competition tier (World Cup 60 ... friendly 20).

Every match is scored *before* it updates the ratings, so the stored
``elo1_pre`` / ``elo2_pre`` are strictly pre-kickoff (no leakage). Friendlies
are included (low K) because they carry real signal about squad form.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

INIT_ELO = 1500.0
HOME_ADV = 100.0


def _k_importance(tournament: str) -> float:
    t = tournament.lower()
    if t == "fifa world cup":
        return 60.0
    if "world cup qualification" in t:
        return 40.0
    if any(x in t for x in ("uefa euro", "copa américa", "copa america",
                            "african cup of nations", "afc asian cup",
                            "gold cup", "confederations")):
        # continental finals tournaments (not their qualifiers)
        return 50.0 if "qualification" not in t else 40.0
    if "qualification" in t or "nations league" in t:
        return 40.0
    if t == "friendly":
        return 20.0
    return 30.0


def _goal_mult(margin: int) -> float:
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def compute_elo(results: pd.DataFrame) -> pd.DataFrame:
    """Return ``results`` with pre-match Elo columns for both teams.

    Adds ``elo1_pre``, ``elo2_pre``, ``elo_diff`` (team1 - team2, including the
    home-advantage adjustment used by the model) and ``elo_prob1`` (Elo win
    expectancy for team1, a calibrated-ish scalar baseline).
    """
    rating: dict[str, float] = defaultdict(lambda: INIT_ELO)
    e1_pre, e2_pre, ediff, eprob = [], [], [], []

    for row in results.itertuples(index=False):
        t1, t2 = row.team1, row.team2
        r1, r2 = rating[t1], rating[t2]
        adv = 0.0 if getattr(row, "neutral", True) else HOME_ADV
        dr = (r1 + adv) - r2
        we1 = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)

        e1_pre.append(r1)
        e2_pre.append(r2)
        ediff.append(dr)
        eprob.append(we1)

        # actual score for team1
        if row.home_score > row.away_score:
            w1 = 1.0
        elif row.home_score < row.away_score:
            w1 = 0.0
        else:
            w1 = 0.5
        margin = int(abs(row.home_score - row.away_score))
        k = _k_importance(row.tournament) * _goal_mult(margin)
        delta = k * (w1 - we1)
        rating[t1] = r1 + delta
        rating[t2] = r2 - delta

    out = results.copy()
    out["elo1_pre"] = e1_pre
    out["elo2_pre"] = e2_pre
    out["elo_diff"] = ediff
    out["elo_prob1"] = eprob
    return out


def ratings_as_of(results_with_elo: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    """Each team's most recent pre-match rating strictly before ``date``.

    Used to attach pre-tournament Elo to World Cup fixtures and to seed the
    bracket simulation. Returns INIT_ELO for teams with no prior match.
    """
    prior = results_with_elo[results_with_elo["date"] < date]
    latest: dict[str, float] = {}
    # recompute a running rating over the prior slice; the last value written
    # per team is its rating going into ``date``.
    rating: dict[str, float] = defaultdict(lambda: INIT_ELO)
    for row in prior.itertuples(index=False):
        r1, r2 = rating[row.team1], rating[row.team2]
        adv = 0.0 if getattr(row, "neutral", True) else HOME_ADV
        dr = (r1 + adv) - r2
        we1 = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        if row.home_score > row.away_score:
            w1 = 1.0
        elif row.home_score < row.away_score:
            w1 = 0.0
        else:
            w1 = 0.5
        margin = int(abs(row.home_score - row.away_score))
        k = _k_importance(row.tournament) * _goal_mult(margin)
        delta = k * (w1 - we1)
        rating[row.team1] = r1 + delta
        rating[row.team2] = r2 - delta
        latest[row.team1] = rating[row.team1]
        latest[row.team2] = rating[row.team2]
    return latest
