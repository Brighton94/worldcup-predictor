# World Cup Model

![FIFA World Cup 2026 predicted knockout bracket from the confirmed Round of 32, with Monte-Carlo title probabilities](_processed_outputs/worldcup/knockouts_2026.png)

A calibrated machine-learning model for international football tournaments. It
rates national teams from forward-in-time **Elo** and **EA FC / FIFA squad
strength**, turns each match into home/draw/away probabilities, and runs a
**Monte-Carlo** simulation of the tournament to produce round-by-round
advancement and title odds. It backtests on the 2018 and 2022 World Cups,
benchmarks several classifiers (logistic regression, random forest, XGBoost,
SVM, kNN), and forecasts the 48-team 2026 bracket, updating live as matches are
played.

The guiding principle is **calibration over accuracy**: a model that is right
55% of the time with trustworthy probabilities is more useful than one that is
right 58% of the time but overconfident. Every metric is reported on a
temporally-correct split (train on the past, test on a later tournament), and no
feature uses information that would not exist 60 seconds before kickoff.

## How the 2026 forecast did

Scored against the real tournament, the model correctly called:

- 13 of the 16 Round-of-16 teams
- 5 of the 8 quarter-finalists
- 4 of the 4 semi-finalists
- both finalists (Spain and Argentina)

## What is inside

| Path | Contents |
| --- | --- |
| `src/worldcup/` | The engine: Elo, features, model, simulation, squads, live update, poster. |
| `src/worldcup/model_zoo.py` | Alternative classifiers (RF, XGBoost, GBM, SVM, kNN) and the temporal backtest harness. |
| `src/worldcup/knockouts_2026.py` | Forecast seeded from the confirmed Round of 32, simulating the rest. |
| `src/analysis/evaluation.py` | Log-loss, Brier, RPS, accuracy, and the baseline probabilities. |
| `notebooks/01_EDA.ipynb` | Exploratory analysis: what the match, rating, and squad data contain. |
| `notebooks/02_hypothesis_tests.ipynb` | Every hypothesis tested, with a null, an alpha = 0.05 test, and a verdict. |
| `notebooks/03_live_api_tests.ipynb` | Live API check: results, fixtures, and the fields available to use. |
| `notebooks/04_worldcup_report.ipynb` | The end-to-end modelling report. |
| `notebooks/05_xg_ratings.ipynb` | StatsBomb expected-goals ratings and an optional xG-blended Elo. |
| `notebooks/06_model_comparison.ipynb` | Logistic regression vs RF, XGBoost, GBM, SVM, kNN on held-out World Cups. |
| `.devcontainer/` | Docker dev container (Python 3.11 + cairo) for a one-click environment. |
| `tests/test_worldcup.py` | Deterministic unit tests for the engine. |
| `_processed_outputs/worldcup/` | Generated results: metrics, predictions, simulations, posters. |
| `data/raw/` | Match history, World Cup data, confirmed squads, flags. |

## How the model works

1. **Elo** (`elo.py`) rates every nation by replaying all international results
   forward in time, with the K-factor scaled by competition importance, a
   goal-difference multiplier, and a host-only home advantage.
2. **Squad strength** (`features.py`, `team_strength.py`, `squads.py`) summarises
   each nation's EA FC / FIFA ratings as a matchday 16 (best XI plus five impact
   subs) alongside goalkeeper, defence, midfield, and attack lines. For 2026 the
   confirmed 26-man squads are matched to EA FC 26 with name-collision guards, so
   a missing star is dropped or aliased rather than replaced by a low-rated
   namesake; nations EA FC barely covers fall back to a nationality-pool proxy.
3. **Model** (`model.py`, `model_zoo.py`) fits a calibrated multinomial logistic
   regression on the rating differences, symmetrised so orientation carries no
   signal. A model comparison (`model_zoo.py`, notebook 06) benchmarks it against
   a random forest, XGBoost, gradient boosting, an SVM, and kNN; the regularised
   random forest gives the best held-out log-loss and drives the knockout
   bracket, while XGBoost overfits this small dataset.
4. **Simulation** (`simulate.py`, `worldcup2026.py`, `knockouts_2026.py`)
   Monte-Carlos the tournament to produce title odds. Before the groups it
   simulates the full 48-team bracket (FIFA Annex C third-place allocation); once
   the Round of 32 is set, `knockouts_2026.py` seeds from the real draw and
   simulates only the knockouts.
5. **Live update** (`live.py`) folds played 2026 results into Elo, locks the real
   group points, and re-forecasts the remaining bracket.

## Main commands

Set up the environment, then run any of the targets below.

```bash
# Option A - Docker dev container (recommended): open the folder in VS Code and
# choose "Reopen in Container". Everything below is then ready to run.

# Option B - local virtualenv:
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

```bash
# Forecasting and backtests
python -m src.worldcup.run            # backtest the 2018 and 2022 World Cups
python -m src.worldcup.run_2026       # forecast the full 48-team 2026 bracket
python -m src.worldcup.knockouts_2026 # forecast from the confirmed Round of 32 (headline + surprise brackets)
python -m src.worldcup.live           # fold in played 2026 results, re-forecast

# Checks
pytest                                 # run the unit tests
```

```bash
# Notebooks (run top-to-bottom)
jupyter notebook notebooks/03_live_api_tests.ipynb     # is the API up; results and fixtures
jupyter notebook notebooks/01_EDA.ipynb                # what the data contains
jupyter notebook notebooks/02_hypothesis_tests.ipynb   # which features worked, with tests
jupyter notebook notebooks/06_model_comparison.ipynb   # logreg vs RF, XGBoost, SVM, kNN
```

The live update and the API-check notebook need `FOOTBALL_DATA_API_KEY` in `.env`
(copy `.env.example`); without it they fall back to the cached snapshot in
`data/raw/footballdata_api/`.

## Data

Match history (martj42 internationals, jfjelstul World Cup data), the confirmed
2026 squads (football-data.org), StatsBomb open-data match xG
(`data/raw/statsbomb/`), and country flags ship in `data/raw/`. The EA
FC / FIFA player ratings (`data/raw/fifa/players_*.csv`) are **not** committed
because they are large and Kaggle-licensed; see `data/raw/fifa/README.md` for how
to supply them. API keys for the live update go in `.env` (template in
`.env.example`).
