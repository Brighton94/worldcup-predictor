"""Continental-tournament signal: form features + the correlation study."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import load_intl_results, load_wc_matches, load_tournaments
from .elo import compute_elo, ratings_as_of

PTS = {"win": 3.0, "draw": 1.0, "loss": 0.0}


# team-centric helpers over the international-results table


def _team_matches(results: pd.DataFrame, team: str, before: pd.Timestamp) -> pd.DataFrame:
    """Matches involving ``team`` strictly before ``before``, team-centric."""
    mask = ((results["team1"] == team) | (results["team2"] == team)) & (results["date"] < before)
    g = results[mask].copy()
    is_home = g["team1"] == team
    g["gf"] = np.where(is_home, g["home_score"], g["away_score"])
    g["ga"] = np.where(is_home, g["away_score"], g["home_score"])
    g["pts"] = np.where(g["gf"] > g["ga"], 3.0, np.where(g["gf"] == g["ga"], 1.0, 0.0))
    return g


def recent_competitive_ppg(results, team, before, months=24) -> tuple[float, int]:
    g = _team_matches(results, team, before)
    g = g[(~g["is_friendly"]) & (g["date"] >= before - pd.DateOffset(months=months))]
    if g.empty:
        return (np.nan, 0)
    return (float(g["pts"].mean()), len(g))


def last_continental_ppg(results, team, before, within_years=4) -> tuple[float, int]:
    """Performance in the team's most recent continental final tournament."""
    cont = results[results["tournament"].isin(C.CONTINENTAL_TOURNAMENTS)]
    g = _team_matches(cont, team, before)
    g = g[g["date"] >= before - pd.DateOffset(years=within_years)]
    if g.empty:
        return (np.nan, 0)
    # restrict to the single most recent edition (matches within 60 days)
    last_date = g["date"].max()
    g = g[g["date"] >= last_date - pd.Timedelta(days=60)]
    return (float(g["pts"].mean()), len(g))


def team_form_features(team: str, before: pd.Timestamp,
                       results: pd.DataFrame | None = None) -> dict:
    """Pre-tournament form features for one team as of ``before``."""
    if results is None:
        results = load_intl_results()
    comp_ppg, comp_n = recent_competitive_ppg(results, team, before)
    cont_ppg, cont_n = last_continental_ppg(results, team, before)
    return {
        "comp_ppg": comp_ppg,
        "comp_n": comp_n,
        "cont_ppg": cont_ppg,
        "cont_played": int(cont_n > 0),
        "cont_deeprun": int(cont_n >= 5),  # >=5 matches implies a knockout run
    }


# correlation study


def cross_tournament_panel(min_year: int = 1998) -> pd.DataFrame:
    """One row per (World Cup, team): WC performance vs prior continental form."""
    results = load_intl_results()
    results_elo = compute_elo(results)
    wc = load_wc_matches()
    tours = load_tournaments().set_index("tournament_id")

    # team-centric WC results
    rows = []
    for tid, g in wc.groupby("tournament_id"):
        year = int(tours.loc[tid, "year"]) if tid in tours.index else None
        if year is None or year < min_year:
            continue
        start = tours.loc[tid, "start_date"]
        elo_at_start = ratings_as_of(results_elo, start)
        teams = set(g["home_team_name"]) | set(g["away_team_name"])
        for team in teams:
            tm = g[(g["home_team_name"] == team) | (g["away_team_name"] == team)]
            is_home = tm["home_team_name"] == team
            res = np.where(is_home, tm["result"], tm["result"].map({"H": "A", "A": "H", "D": "D"}))
            pts = pd.Series(res).map({"H": 3.0, "D": 1.0, "A": 0.0}).mean()
            cont_ppg, cont_n = last_continental_ppg(results, team, start)
            comp_ppg, _ = recent_competitive_ppg(results, team, start)
            rows.append({
                "tournament_id": tid, "year": year, "team": team,
                "wc_ppg": float(pts), "wc_matches": len(tm),
                "cont_ppg": cont_ppg, "cont_n": cont_n, "comp_ppg": comp_ppg,
                "pre_elo": elo_at_start.get(team, 1500.0),
            })
    return pd.DataFrame(rows)


def analyse_correlation(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Correlations between pre-WC form and WC points-per-game."""
    if panel is None:
        panel = cross_tournament_panel()
    out = []
    for col, label in [("cont_ppg", "continental campaign ppg"),
                       ("comp_ppg", "recent competitive ppg (24m)"),
                       ("pre_elo", "pre-tournament Elo (benchmark)")]:
        sub = panel.dropna(subset=[col, "wc_ppg"])
        if len(sub) < 10:
            continue
        out.append({
            "predictor": label,
            "n": len(sub),
            "pearson": float(sub[col].corr(sub["wc_ppg"], method="pearson")),
            "spearman": float(sub[col].corr(sub["wc_ppg"], method="spearman")),
        })
    return pd.DataFrame(out)
