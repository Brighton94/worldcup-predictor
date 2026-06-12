# World Cup prediction module

A self-contained international-tournament pipeline that lives alongside the
club-league engine. It predicts World Cup match outcomes (regulation 1X2),
simulates the bracket for round-advancement and title odds, and studies whether
continental-tournament form predicts World Cup performance.

It reuses only `src/analysis/evaluation.py` (scoring rules) from the rest of the
repo. Class encoding follows the project convention: `H = 0, D = 1, A = 2`.

## Data sources

None of these are committed (`data/raw/` is gitignored). To reproduce, place the
files as below.

| Source | Path | Provides |
|---|---|---|
| [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) | `data/raw/worldcup/*.csv` | World Cup matches, groups, tournaments, bracket structure |
| [martj42/international_results](https://github.com/martj42/international_results) | `data/raw/international/{results,shootouts}.csv` | ~49k international matches 1872-2026 (Elo, continental form, training backbone) |
| sofifa FIFA editions ([stefanoleone992](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset) and mirrors) | `data/raw/fifa/players_17.csv` ... `players_23.csv` | Player ratings with nationality and positions |

FIFA editions currently present are 17-23 (FIFA 23 from the open
[jsulz/FIFA23](https://huggingface.co/datasets/jsulz/FIFA23) mirror). Editions
14-16 can be added the same way. `config.active_edition` picks the latest edition
released on or before a given date, so FIFA 23 (released Sep 2022) is now the
active edition for both the 2022 and 2026 tournaments.

**EA FC 24/25/26 are not yet sourced.** They exist only behind Kaggle
authentication (no open full-coverage mirror), and EA FC 26 itself omits several
national teams (e.g. Brazil) for licensing reasons. To add them, drop the
sofifa-schema CSVs in as `players_24.csv`/`players_25.csv`/`players_26.csv` (an
`overall` column plus `nationality_name`) and add their release dates to
`config.FIFA_RELEASE`. The loader skips any file not in the sofifa schema, so a
partial/incompatible export left in the folder is ignored rather than used.

## Design (and the leakage rules it respects)

- **Temporal split, never random.** Train on competitive internationals before
  the target World Cup; test on the tournament itself.
- **Leak-free target.** World Cup scores in the source include extra time. The
  regulation (90-minute) result is recovered exactly: a knockout that went to
  extra time was level after 90, i.e. a draw. Shootout matches are dropped from
  the training backbone (their regulation label is ambiguous).
- **Pre-kickoff features only.** Elo is snapshotted *before* each match updates
  it. FIFA strength uses the edition active at the match date (or at the
  tournament start for the test set). Continental form is time-bounded to before
  kickoff.
- **Orientation invariance.** World Cup venues are neutral and the source
  "home/away" labelling is an artefact, so features are symmetric differences
  (`team1 - team2`) plus a signed `home_field` host term, and the training set is
  mirrored so the model cannot exploit listing order.
- **Why FIFA ratings give a real training set.** FIFA editions cover every
  nation, so *every* international match in 2017-2022 can be enriched with both
  teams' squad strength — thousands of training rows, not just the 64 World Cup
  matches.

## Usage

```bash
# from the repo root
python -m src.worldcup.run                 # 2018 and 2022, 20k sims
python -m src.worldcup.run --year 2022 --sims 50000
pytest tests/test_worldcup.py
```

Outputs are written to `_processed_outputs/worldcup/` (metrics, per-match
predictions, simulation tables, cross-tournament study). The report notebook is
`notebooks/worldcup_report.ipynb`.

## Results (held-out tournaments)

The full model (Elo + FIFA strength) beats both baselines on log-loss and Brier
in both held-out tournaments; the eventual champions ranked among the top teams
in the simulation.

| Tournament | Model | Log-loss | Brier | Accuracy |
|---|---|---|---|---|
| 2018 | full (Elo + FIFA) | 0.951 | 0.565 | 0.563 |
| 2018 | Elo-only | 0.987 | 0.584 | 0.516 |
| 2018 | class-prior | 1.105 | 0.672 | 0.391 |
| 2022 | full (Elo + FIFA) | 0.978 | 0.575 | 0.578 |
| 2022 | Elo-only | 1.049 | 0.614 | 0.484 |
| 2022 | class-prior | 1.063 | 0.643 | 0.453 |

Cross-tournament study: raw continental-tournament points-per-game barely
correlates with World Cup points-per-game (Pearson approx -0.02), whereas
pre-tournament Elo correlates strongly (approx 0.52). The cross-tournament
signal is real but already absorbed into team strength, so it is not added as a
separate feature.

## Modules

`config` paths/aliases/cycle map · `data` loaders · `elo` World-Football Elo ·
`team_strength` FIFA squad aggregation · `features` match matrix ·
`cross_tournament` form features + correlation study · `model` train/calibrate/
baselines · `simulate` Monte-Carlo bracket · `run` end-to-end CLI.

## Limitations / next steps

- 2018 cycle uses only FIFA 17-18 (14-16 not yet sourced); 2022 uses 19-22.
- Qatar has no FIFA representation (domestic league absent from the game) and is
  imputed with a thin-squad profile; its Elo is real.
- Bookmaker odds are not available for historical World Cups in these free
  sources, so Elo serves as the strong baseline in place of market-implied
  probabilities.
- To predict the 2026 World Cup, add FIFA 23-26 ratings, register the cycle, and
  supply the group draw once published.
