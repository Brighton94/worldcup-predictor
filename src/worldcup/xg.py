"""StatsBomb open-data match xG, keyed by date and the two teams (canonical names)."""

from __future__ import annotations

import pandas as pd

from . import config as C

XG_FILE = C.RAW / "statsbomb" / "statsbomb_intl_xg.csv"


def load_xg_lookup() -> dict:
    """{(normalised date, frozenset{team_a, team_b}): {team: match xG}} from StatsBomb tournaments."""
    if not XG_FILE.exists():
        return {}
    d = pd.read_csv(XG_FILE)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    lookup: dict = {}
    for r in d.itertuples(index=False):
        a, b = C.canon_intl(r.home), C.canon_intl(r.away)
        lookup[(r.date, frozenset({a, b}))] = {a: float(r.home_xg), b: float(r.away_xg)}
    return lookup
