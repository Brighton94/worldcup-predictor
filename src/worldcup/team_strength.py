"""Per-nation squad strength from FIFA player ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import load_fifa_players

SQUAD_N = 23  # nominal squad size

_POS_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "ST": "ATT", "CF": "ATT", "LW": "ATT", "RW": "ATT", "LF": "ATT", "RF": "ATT",
}


def _primary_group(player_positions: str) -> str:
    first = str(player_positions).split(",")[0].strip().upper()
    return _POS_GROUP.get(first, "MID")


def _edition_strength(players: pd.DataFrame) -> pd.DataFrame:
    """Strength features per nationality for one FIFA edition."""
    df = players.copy()
    df["group"] = df["player_positions"].map(_primary_group)
    rows = []
    for nat, g in df.groupby("nationality"):
        g = g.sort_values("overall", ascending=False)
        squad = g.head(SQUAD_N)
        if len(squad) < 11:
            continue
        best11 = g.head(11)
        def grp_mean(group: str, n: int) -> float:
            vals = g[g["group"] == group]["overall"].head(n)
            return float(vals.mean()) if len(vals) else np.nan
        rows.append({
            "nationality": nat,
            "ovr_top23": float(squad["overall"].mean()),
            "ovr_top11": float(best11["overall"].mean()),
            "ovr_max": float(g["overall"].max()),
            "ovr_top3": float(g["overall"].head(3).mean()),
            "gk": float(g[g["group"] == "GK"]["overall"].max()) if (g["group"] == "GK").any() else np.nan,
            "def_top": grp_mean("DEF", 5),
            "mid_top": grp_mean("MID", 4),
            "att_top": grp_mean("ATT", 3),
            "depth80": int((squad["overall"] >= 80).sum()),
            "age_top23": float(squad["age"].mean()),
        })
    return pd.DataFrame(rows).set_index("nationality")


def build_team_strength(cycle_year: int) -> pd.DataFrame:
    """Recency-weighted squad strength per nation for a World Cup cycle."""
    editions = C.CYCLE_EDITIONS[cycle_year]
    fifa = load_fifa_players()
    fifa = fifa[fifa["edition"].isin(editions)]

    base = min(editions)
    weights = {e: (e - base + 1) for e in editions}
    per_edition = {e: _edition_strength(fifa[fifa["edition"] == e]) for e in editions}

    feat_cols = ["ovr_top23", "ovr_top11", "ovr_max", "ovr_top3", "gk",
                 "def_top", "mid_top", "att_top", "depth80", "age_top23"]
    nations = sorted(set().union(*[d.index for d in per_edition.values()]))
    out = pd.DataFrame(index=pd.Index(nations, name="nationality"), columns=feat_cols, dtype=float)
    for nat in nations:
        for col in feat_cols:
            num, den = 0.0, 0.0
            for e in editions:
                d = per_edition[e]
                if nat in d.index and not pd.isna(d.loc[nat, col]):
                    num += weights[e] * d.loc[nat, col]
                    den += weights[e]
            out.loc[nat, col] = num / den if den else np.nan
    out["cycle_year"] = cycle_year
    return out.reset_index()
