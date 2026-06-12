"""Data loaders with a single canonical country naming.

Three sources, joined on canonical country names:

* World Cup structured data (jfjelstul) -- defines the matches to predict, the
  bracket structure, and exact 90-minute labels (the extra-time flag lets us
  recover the regulation result: a knockout that went to extra time was level
  after 90, i.e. a draw).
* International results (martj42) -- ~49k matches 1872-2026, the training
  backbone and the source for Elo and continental-tournament form. Scores
  include extra time, so penalty-shootout matches (separate file) are dropped
  from clean-label training.
* FIFA player ratings (sofifa) -- squad-strength enrichment per nation/edition.

Target encoding (regulation, 90 min + stoppage): H = 0, D = 1, A = 2.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import config as C

RESULT_TO_IDX = {"H": 0, "D": 1, "A": 2}


# ── World Cup structured data ────────────────────────────────────────


def load_wc_matches(mens_only: bool = True) -> pd.DataFrame:
    """World Cup matches with a leak-free regulation (90-min) result.

    A knockout match with ``extra_time == 1`` was level after 90 minutes, so
    its regulation result is a draw regardless of who advanced. Group matches
    never go to extra time, so their recorded result is already regulation.
    """
    m = pd.read_csv(C.WC_RAW / "matches.csv", low_memory=False)
    if mens_only:
        m = m[m["tournament_name"].str.contains("Men's")].copy()
    m = m[m["replay"] == 0].copy()  # drop replays of replayed fixtures
    m["match_date"] = pd.to_datetime(m["match_date"])

    def regulation_result(row) -> str:
        if row["extra_time"] == 1:  # level after 90 -> regulation draw
            return "D"
        if row["home_team_win"] == 1:
            return "H"
        if row["away_team_win"] == 1:
            return "A"
        return "D"

    m["result"] = m.apply(regulation_result, axis=1)
    m["y"] = m["result"].map(RESULT_TO_IDX)
    return m


def load_tournaments() -> pd.DataFrame:
    t = pd.read_csv(C.WC_RAW / "tournaments.csv")
    t = t[t["tournament_name"].str.contains("Men's")].copy()
    t["start_date"] = pd.to_datetime(t["start_date"])
    t["end_date"] = pd.to_datetime(t["end_date"])
    return t


def host_country(tournament_id: int | str) -> str:
    """Canonical host-country name for a World Cup id (e.g. 'WC-2022')."""
    t = load_tournaments()
    row = t[t["tournament_id"] == tournament_id]
    if row.empty:
        return ""
    return str(row.iloc[0]["host_country"])


# ── International results (training backbone, Elo, form) ──────────────


def load_intl_results() -> pd.DataFrame:
    """All international matches, canonicalised, with clean-label flags.

    Columns added: ``team1``/``team2`` (canonical), ``is_friendly``,
    ``is_shootout`` (went to penalties -> ambiguous regulation label),
    ``result`` (H/D/A from the recorded score), ``y``.
    """
    r = pd.read_csv(C.INTL_RAW / "results.csv")
    r["date"] = pd.to_datetime(r["date"])
    r = r.dropna(subset=["home_score", "away_score"]).copy()
    r["team1"] = r["home_team"].map(C.canon_intl)
    r["team2"] = r["away_team"].map(C.canon_intl)
    r["is_friendly"] = r["tournament"].eq("Friendly")
    r["neutral"] = r["neutral"].astype(str).str.upper().eq("TRUE")

    # penalty-shootout matches: level after extra time -> regulation label
    # is ambiguous from the score alone; flag them for exclusion.
    s = pd.read_csv(C.INTL_RAW / "shootouts.csv")
    s["date"] = pd.to_datetime(s["date"])
    shootout_keys = set(
        zip(s["date"], s["home_team"].map(C.canon_intl), s["away_team"].map(C.canon_intl))
    )
    r["is_shootout"] = [
        (d, a, b) in shootout_keys for d, a, b in zip(r["date"], r["team1"], r["team2"])
    ]

    def res(row) -> str:
        if row["home_score"] > row["away_score"]:
            return "H"
        if row["home_score"] < row["away_score"]:
            return "A"
        return "D"

    r["result"] = r.apply(res, axis=1)
    r["y"] = r["result"].map(RESULT_TO_IDX)
    return r.sort_values("date").reset_index(drop=True)


# ── FIFA player ratings ──────────────────────────────────────────────

_FIFA_COLS = ["short_name", "overall", "potential", "age", "player_positions"]

# Minimum distinct nationalities for a file to count as full national coverage.
# Filters out club-only or top-leagues-only exports (e.g. a 30-nation EA FC 24
# file) that would distort national-team strength.
_MIN_NATIONS = 75

# Map common column-name variants (lowercased) onto the canonical schema, so
# whichever EA FC / FIFA export is dropped in is understood without edits.
_COL_VARIANTS = {
    "nationality": ["nationality_name", "nationality", "country_name", "nation", "country"],
    "overall": ["overall", "overall_rating", "ova", "ovr", "rating"],
    "short_name": ["short_name", "name", "long_name", "player", "player_name"],
    "long_name": ["long_name", "short_name", "name"],
    "player_positions": ["player_positions", "positions", "position", "best_position"],
    "age": ["age"],
    "potential": ["potential", "potential_rating", "pot", "best_overall"],
}


def _normalise_fifa(df: pd.DataFrame) -> pd.DataFrame | None:
    """Rename a raw player export to the canonical schema, or None if unusable."""
    lower = {c.lower(): c for c in df.columns}
    chosen = {}
    for canon, variants in _COL_VARIANTS.items():
        for v in variants:
            if v in lower:
                chosen[canon] = lower[v]
                break
    if "overall" not in chosen or "nationality" not in chosen:
        return None  # not a player-ratings table with nationality
    out = pd.DataFrame()
    for canon, src in chosen.items():
        out[canon] = df[src]
    for opt in ("potential", "age", "player_positions", "short_name", "long_name"):
        if opt not in out.columns:
            out[opt] = pd.NA
    out["overall"] = pd.to_numeric(out["overall"], errors="coerce")
    return out.dropna(subset=["overall", "nationality"])


@lru_cache(maxsize=1)
def load_fifa_players() -> pd.DataFrame:
    """All FIFA/EA FC editions stacked, with canonical ``nationality``/``edition``.

    Robust to the different column namings used by the various sofifa and EA
    Sports FC exports, and to latin-1 encoding. Files that are not player-rating
    tables, or that lack full national coverage (``_MIN_NATIONS``), are skipped
    so a partial export cannot corrupt the strength tables.
    """
    frames = []
    for path in sorted(C.FIFA_RAW.glob("players_*.csv")):
        year = 2000 + int(path.stem.split("_")[-1])
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        sub = _normalise_fifa(df)
        if sub is None:
            continue
        sub["nationality"] = sub["nationality"].map(C.canon_fifa)
        if sub["nationality"].nunique() < _MIN_NATIONS:
            continue  # club-only / partial export
        sub["edition"] = year
        frames.append(sub[_FIFA_COLS + ["nationality", "edition"]])
    return pd.concat(frames, ignore_index=True)


def available_editions() -> list[int]:
    """Editions actually present with full national coverage (sorted)."""
    return sorted(load_fifa_players()["edition"].unique())
