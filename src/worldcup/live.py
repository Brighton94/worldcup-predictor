"""Live in-tournament update: fold played 2026 results into Elo + the bracket.

As World Cup matches finish, football-data.org reports the scores. This module:
1. fetches the played 2026 results (and caches them),
2. appends them to the international history and recomputes Elo (so ratings
   reflect what's happened),
3. uses the confirmed-squad strength table (``squads``) instead of the
   nationality-pool proxy,
4. locks the played group games into the forecast (real points, not simulated)
   and re-runs the Monte-Carlo bracket on what's left,
5. writes updated outputs and renders a "LIVE" bracket poster.

    python -m src.worldcup.live              # refresh results and re-forecast
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import urllib.request

import cairosvg
import pandas as pd

from . import config as C
from .data import load_intl_results
from .elo import compute_elo
from .squads import canon, confirmed_strength_table
from .worldcup2026 import build_2026, HOSTS_2026
from .run_2026 import _bracket_rows
from .poster import build_poster

WC_FILE = C.RAW / "footballdata_api" / "WC_2026_matches.json"
_PTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def fetch_results(refresh: bool = True) -> list[dict]:
    """Finished 2026 WC matches (canonical names, scores, group)."""
    if refresh or not WC_FILE.exists():
        key = os.getenv("FOOTBALL_DATA_API_KEY", "")
        try:
            req = urllib.request.Request(
                "https://api.football-data.org/v4/competitions/WC/matches",
                headers={"X-Auth-Token": key})
            data = json.load(urllib.request.urlopen(req, timeout=40))
            json.dump(data, open(WC_FILE, "w"))
        except Exception as e:  # offline / rate-limited: fall back to cache
            print(f"  (using cached results: {e})")
            data = json.load(open(WC_FILE))
    else:
        data = json.load(open(WC_FILE))
    out = []
    for m in data["matches"]:
        if m["status"] != "FINISHED":
            continue
        ft = m["score"]["fullTime"]
        out.append({
            "date": m["utcDate"][:10],
            "home": canon(m["homeTeam"]["name"]), "away": canon(m["awayTeam"]["name"]),
            "hg": int(ft["home"]), "ag": int(ft["away"]),
            "group": (m.get("group") or "").replace("GROUP_", "") or None,
            "stage": m["stage"],
        })
    return out


def _result_letter(hg, ag):
    return "H" if hg > ag else ("D" if hg == ag else "A")


def updated_elo(finished: list[dict]):
    """Elo results including the played WC games (high-weight FIFA World Cup)."""
    base = load_intl_results()
    rows = []
    for m in finished:
        rows.append({
            "date": pd.Timestamp(m["date"]), "team1": m["home"], "team2": m["away"],
            "home_score": m["hg"], "away_score": m["ag"],
            "neutral": m["home"] not in HOSTS_2026,   # hosts get home advantage
            "tournament": "FIFA World Cup", "is_friendly": False, "is_shootout": False,
        })
    combined = pd.concat([base, pd.DataFrame(rows)], ignore_index=True) if rows else base
    combined = combined.sort_values("date").reset_index(drop=True)
    return compute_elo(combined)


def played_group_games(finished: list[dict]) -> dict:
    """frozenset({home, away}) -> {team: points} for finished GROUP games."""
    played = {}
    for m in finished:
        if m["stage"] != "GROUP_STAGE":
            continue
        ph, pa = _PTS[_result_letter(m["hg"], m["ag"])]
        played[frozenset({m["home"], m["away"]})] = {m["home"]: ph, m["away"]: pa}
    return played


def run(n_sims: int = 20000, date_label: str = None):
    finished = fetch_results(refresh=True)
    played = played_group_games(finished)
    results_elo = updated_elo(finished)
    squad_tbl, _ = confirmed_strength_table()
    as_of = pd.Timestamp(_dt.date.today()) + pd.Timedelta(days=1)

    res = build_2026(n_sims=n_sims, squad_table=squad_tbl, results_elo=results_elo,
                     as_of_date=as_of, played=played)

    _bracket_rows(res["bracket"]).to_csv(C.OUT / "live_predictions_2026_bracket.csv", index=False)
    res["sim"].to_csv(C.OUT / "live_simulation.csv", index=False)

    n_played = len(finished)
    sub = (f"A machine-learning forecast, updated after {n_played} played "
           f"match(es). Bracket per FIFA Annex C.")
    svg = build_poster(date_label or _dt.date.today().strftime("%-d %B %Y"), "@brighton_nkomo_",
                       bracket_csv=C.OUT / "live_predictions_2026_bracket.csv",
                       sim_csv=C.OUT / "live_simulation.csv",
                       title_text="World Cup 2026: LIVE Predicted Bracket",
                       subtitle_text=sub, pill_note="")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                     write_to=str(C.OUT / "live_bracket_2026.png"), output_width=2800)
    return res, finished


def main():
    res, finished = run()
    print(f"played matches folded in: {len(finished)}")
    for m in finished:
        print(f"  {m['home']} {m['hg']}-{m['ag']} {m['away']}  ({m['stage']})")
    b = res["bracket"]
    print(f"\nupdated champion: {b[104][2]} | final: {b[101][2]} vs {b[102][2]}")
    print("\nTitle probabilities (top 8):")
    print(res["sim"].head(8).to_string(index=False))
    print(f"\nWrote live_bracket_2026.png + live_*.csv to {C.OUT}")


if __name__ == "__main__":
    main()
