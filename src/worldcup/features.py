"""Assemble the match-level training matrix."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from .data import load_intl_results, load_wc_matches, load_tournaments, RESULT_TO_IDX
from .elo import compute_elo, ratings_as_of
from .team_strength import _edition_strength
from .data import load_fifa_players

# strength columns differenced into features
_STRENGTH = ["ovr_top23", "ovr_top11", "ovr_top3", "gk", "def_top", "mid_top",
             "att_top", "depth80"]
FEATURES = ["d_elo", "home_field"] + [f"d_{c}" for c in _STRENGTH]


@lru_cache(maxsize=None)
def _edition_table(edition: int) -> pd.DataFrame:
    """Strength table for a single FIFA edition, indexed by canonical nation."""
    fifa = load_fifa_players()
    tbl = _edition_strength(fifa[fifa["edition"] == edition])
    return tbl.fillna(tbl.median(numeric_only=True))


def edition_table_filled(edition: int) -> pd.DataFrame:
    """Strength table for ``edition``, backfilling nations it omits."""
    from .data import available_editions

    base = _edition_table(edition)
    have = set(base.index)
    for e in sorted((x for x in available_editions() if x < edition), reverse=True):
        t = _edition_table(e)
        missing = [n for n in t.index if n not in have]
        if missing:
            base = pd.concat([base, t.loc[missing]])
            have.update(missing)
    return base


def strength_vector(tbl: pd.DataFrame, team: str) -> pd.Series:
    """Strength row for ``team``, imputing a thin-squad profile if absent."""
    if team in tbl.index:
        return tbl.loc[team]
    return tbl[_STRENGTH].quantile(0.10)


def _diff_row(s1: pd.Series, s2: pd.Series) -> dict:
    return {f"d_{c}": float(s1[c] - s2[c]) for c in _STRENGTH}


def build_backbone(before: pd.Timestamp, results_elo: pd.DataFrame) -> pd.DataFrame:
    """Competitive internationals before ``before`` with full features."""
    df = results_elo[
        (~results_elo["is_friendly"]) & (~results_elo["is_shootout"])
        & (results_elo["date"] < before)
        & (results_elo["date"] >= C.FIFA_RELEASE[min(C.FIFA_RELEASE)])
    ].copy()

    rows = []
    for r in df.itertuples(index=False):
        ed = C.active_edition(r.date)
        if ed is None:
            continue
        tbl = _edition_table(ed)
        if r.team1 not in tbl.index or r.team2 not in tbl.index:
            continue
        s1, s2 = tbl.loc[r.team1], tbl.loc[r.team2]
        if s1[_STRENGTH].isna().any() or s2[_STRENGTH].isna().any():
            continue
        row = {
            "date": r.date, "team1": r.team1, "team2": r.team2,
            "d_elo": r.elo1_pre - r.elo2_pre,
            "home_field": 0.0 if r.neutral else 1.0,
            "y": r.y,
        }
        row.update(_diff_row(s1, s2))
        rows.append(row)
    return pd.DataFrame(rows)


def build_wc_test(wc_year: int, results_elo: pd.DataFrame) -> pd.DataFrame:
    """World Cup fixtures for ``wc_year`` with the same feature representation."""
    tid = f"WC-{wc_year}"
    tours = load_tournaments().set_index("tournament_id")
    start = tours.loc[tid, "start_date"]
    host = str(tours.loc[tid, "host_country"])
    ed = C.active_edition(start)
    tbl = _edition_table(ed)
    elo_at_start = ratings_as_of(results_elo, start)

    wc = load_wc_matches()
    wc = wc[wc["tournament_id"] == tid].copy()

    rows = []
    for r in wc.itertuples(index=False):
        t1, t2 = r.home_team_name, r.away_team_name
        s1, s2 = strength_vector(tbl, t1), strength_vector(tbl, t2)
        hf = 1.0 if t1 == host else (-1.0 if t2 == host else 0.0)
        row = {
            "date": r.match_date, "team1": t1, "team2": t2,
            "stage": r.stage_name, "knockout": int(r.knockout_stage),
            "d_elo": elo_at_start.get(t1, 1500.0) - elo_at_start.get(t2, 1500.0),
            "home_field": hf,
            "y": r.y,
        }
        row.update(_diff_row(s1, s2))
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(wc_year: int) -> dict:
    """Backbone (train) + World Cup (test) for a target tournament."""
    results_elo = compute_elo(load_intl_results())
    tours = load_tournaments().set_index("tournament_id")
    start = tours.loc[f"WC-{wc_year}", "start_date"]
    train = build_backbone(start, results_elo)
    test = build_wc_test(wc_year, results_elo)
    return {"train": train, "test": test, "edition": C.active_edition(start),
            "wc_start": start}
