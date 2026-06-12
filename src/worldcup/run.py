"""End-to-end World Cup pipeline: features -> model -> baselines -> bracket."""

from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from .model import train_eval
from .simulate import simulate
from .cross_tournament import cross_tournament_panel, analyse_correlation


def run_year(year: int, n_sims: int) -> None:
    print(f"\n{'=' * 64}\nWorld Cup {year}\n{'=' * 64}")
    res = train_eval(year)
    print(f"training matches: {res['n_train']}   test matches: {len(res['test'])}")
    print("\nHeld-out tournament metrics (lower log-loss / Brier is better):")
    print(res["results"].to_string(index=False))

    res["results"].to_csv(C.OUT / f"metrics_{year}.csv", index=False)
    cols = ["date", "team1", "team2", "stage", "p_team1", "p_draw", "p_team2", "result", "y"]
    test = res["test"].copy()
    test["result"] = test["y"].map({0: "H", 1: "D", 2: "A"})
    test[[c for c in cols if c in test.columns]].to_csv(
        C.OUT / f"predictions_{year}.csv", index=False)

    sim = simulate(year, res["model"], n_sims=n_sims)
    print(f"\nTitle probabilities (top 8, {n_sims} sims):")
    print(sim.head(8).to_string(index=False))
    actual = {2018: "France", 2022: "Argentina"}.get(year)
    if actual is not None:
        row = sim[sim["team"] == actual]
        if not row.empty:
            print(f"\nActual winner {actual}: modelled title prob = "
                  f"{row.iloc[0]['Champion']:.3f}, reach final = {row.iloc[0]['Final']:.3f}")
    sim.to_csv(C.OUT / f"simulation_{year}.csv", index=False)


def run_cross_tournament() -> None:
    print(f"\n{'=' * 64}\nCross-tournament correlation study\n{'=' * 64}")
    panel = cross_tournament_panel()
    corr = analyse_correlation(panel)
    print(corr.to_string(index=False))
    panel.to_csv(C.OUT / "cross_tournament_panel.csv", index=False)
    corr.to_csv(C.OUT / "cross_tournament.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, choices=[2018, 2022])
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    years = [args.year] if args.year else [2018, 2022]
    for y in years:
        run_year(y, args.sims)
    run_cross_tournament()
    print(f"\nOutputs written to {C.OUT}")


if __name__ == "__main__":
    main()
