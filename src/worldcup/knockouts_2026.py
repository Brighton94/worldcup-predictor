"""Knockout forecast seeded from the confirmed Round of 32, simulating the rest.

The 16 confirmed Round-of-32 matchups are fixed; every later round is simulated
from the model at neutral knockout venues. Edit R32_2026 as results come in.

    python -m src.worldcup.knockouts_2026
"""

from __future__ import annotations

import cairosvg
import numpy as np
import pandas as pd

from . import config as C
from .data import load_intl_results
from .elo import compute_elo, ratings_as_of
from .model import train_eval
from .squads import confirmed_strength_table
from .worldcup2026 import pairwise_probs, R16, QF, SF, FINAL
from .run_2026 import _bracket_rows
from .poster import build_poster

WC_2026_START = pd.Timestamp("2026-06-11")

# Confirmed Round of 32 (official match number -> canonical team names). Edit as results land.
R32_2026 = {
    73: ("South Africa", "Canada"), 74: ("Germany", "Paraguay"), 75: ("Netherlands", "Morocco"),
    76: ("Brazil", "Japan"), 77: ("France", "Sweden"), 78: ("Ivory Coast", "Norway"),
    79: ("Mexico", "Ecuador"), 80: ("England", "Congo DR"),
    81: ("United States", "Bosnia and Herzegovina"), 82: ("Belgium", "Senegal"),
    83: ("Portugal", "Croatia"), 84: ("Spain", "Austria"), 85: ("Switzerland", "Algeria"),
    86: ("Argentina", "Cape Verde"), 87: ("Colombia", "Ghana"), 88: ("Australia", "Egypt"),
}


def simulate_from_r32(r32, model, tbl, elo, n_sims=20000, seed=7):
    """Deterministic bracket + Monte-Carlo title odds, seeded from fixed R32 matchups."""
    teams = sorted({t for m in r32.values() for t in m})
    pw = pairwise_probs(model, teams, tbl, elo, frozenset())   # knockouts are neutral-venue

    def adv(a, b):  # draw -> split by relative win probability
        p = pw[(a, b)]
        return p[0] + 0.5 * p[1]

    bracket = {m: (a, b, a if adv(a, b) >= 0.5 else b) for m, (a, b) in r32.items()}
    for rnd in (R16, QF, SF, FINAL):
        for m, (x, y) in rnd.items():
            a, b = bracket[x][2], bracket[y][2]
            bracket[m] = (a, b, a if adv(a, b) >= 0.5 else b)

    rng = np.random.default_rng(seed)
    cols = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    tally = {t: dict.fromkeys(cols, 0) for t in teams}
    for t in teams:
        tally[t]["R32"] = n_sims
    for _ in range(n_sims):
        win = {}
        for m, (a, b) in r32.items():
            w = a if rng.random() < adv(a, b) else b
            win[m] = w
            tally[w]["R16"] += 1
        for rnd, lab in ((R16, "QF"), (QF, "SF"), (SF, "Final"), (FINAL, "Champion")):
            for m, (x, y) in rnd.items():
                a, b = win[x], win[y]
                w = a if rng.random() < adv(a, b) else b
                win[m] = w
                tally[w][lab] += 1
    sim = (pd.DataFrame(tally).T / n_sims).reset_index().rename(columns={"index": "team"})
    sim = sim.sort_values("Champion", ascending=False).reset_index(drop=True)
    return {"bracket": bracket, "sim": sim}


def run(n_sims=20000, date_label="28 June 2026"):
    model = train_eval(2022)["model"]
    elo = ratings_as_of(compute_elo(load_intl_results()), WC_2026_START)
    tbl, _ = confirmed_strength_table()
    res = simulate_from_r32(R32_2026, model, tbl, elo, n_sims=n_sims)

    _bracket_rows(res["bracket"]).to_csv(C.OUT / "knockouts_2026_bracket.csv", index=False)
    res["sim"].to_csv(C.OUT / "knockouts_2026_simulation.csv", index=False)
    svg = build_poster(date_label, "@brighton_nkomo_",
                       bracket_csv=C.OUT / "knockouts_2026_bracket.csv",
                       sim_csv=C.OUT / "knockouts_2026_simulation.csv",
                       title_text="World Cup 2026: Predicted Knockout Bracket",
                       subtitle_text="A machine-learning forecast from the confirmed Round of 32. Bracket per FIFA.",
                       pill_note="before the knockout matches")
    cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                     write_to=str(C.OUT / "knockouts_2026.png"), output_width=2800)
    return res


def main():
    res = run()
    b = res["bracket"]
    print(f"champion: {b[104][2]} | final: {b[101][2]} vs {b[102][2]}")
    print(res["sim"].head(8)[["team", "Champion"]].to_string(index=False))
    print("wrote knockouts_2026.png + CSVs to", C.OUT)


if __name__ == "__main__":
    main()
