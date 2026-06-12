# World Cup Predictor

A calibrated machine-learning model for the FIFA World Cup. It rates national
teams from forward-in-time **Elo** and **EA FC / FIFA squad strength**, fits a
calibrated multinomial model for home/draw/away, and runs a **Monte-Carlo**
simulation of the tournament to produce round-by-round advancement and title
odds. It backtests on the 2018 and 2022 World Cups and forecasts the 48-team
2026 bracket, updating live as matches are played.

The guiding principle is **calibration over accuracy**: a model that is right
55% of the time with trustworthy probabilities is more useful than one that is
right 58% of the time but overconfident. Every metric is reported on a
temporally-correct split (train on the past, test on a later tournament), and no
feature uses information that would not exist 60 seconds before kickoff.

## What is inside

| Path | Contents |
| --- | --- |
| `src/worldcup/` | The engine: Elo, features, model, simulation, squads, live update, poster. |
| `src/analysis/evaluation.py` | Log-loss, Brier, RPS, accuracy, and the baseline probabilities. |
| `notebooks/01_data_and_hypotheses.ipynb` | Data understanding plus every hypothesis tested, with verdicts. |
| `notebooks/worldcup_report.ipynb` | The end-to-end modelling report. |
| `tests/test_worldcup.py` | Deterministic unit tests for the engine. |
| `_processed_outputs/worldcup/` | Generated results: metrics, predictions, simulations, posters. |
| `data/raw/` | Match history, World Cup data, confirmed squads, flags. |

## How the model works

1. **Elo** (`elo.py`) rates every nation by replaying all international results
   forward in time, with the K-factor scaled by competition importance, a
   goal-difference multiplier, and a host-only home advantage.
2. **Squad strength** (`features.py`, `team_strength.py`, `squads.py`) summarises
   each nation's EA FC / FIFA ratings (top-23, goalkeeper, defence, midfield,
   attack). For 2026 the confirmed 26-man squads are matched to EA FC 26;
   nations EA FC does not cover fall back to a nationality-pool proxy.
3. **Model** (`model.py`) fits a multinomial logistic regression on the rating
   differences, symmetrised so orientation carries no signal, and calibrated
   with sigmoid (Platt) scaling on a temporal tail.
4. **Simulation** (`simulate.py`, `worldcup2026.py`) Monte-Carlos the group stage
   and knockouts (the 2026 bracket follows the official FIFA Annex C
   third-place allocation) to produce title odds.
5. **Live update** (`live.py`) folds played 2026 results into Elo, locks the real
   group points, and re-forecasts the remaining bracket.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.worldcup.run            # backtest 2018 and 2022
python -m src.worldcup.run_2026       # forecast the 2026 bracket
python -m src.worldcup.defense_study  # the defence-wins-titles study
python -m src.worldcup.live           # update the forecast with played results
pytest                                 # run the tests
```

Then open `notebooks/01_data_and_hypotheses.ipynb` for the data analysis and the
record of which features worked.

## Data

Match history (martj42 internationals, jfjelstul World Cup data), the confirmed
2026 squads (football-data.org), and country flags ship in `data/raw/`. The EA
FC / FIFA player ratings (`data/raw/fifa/players_*.csv`) are **not** committed
because they are large and Kaggle-licensed; see `data/raw/fifa/README.md` for how
to supply them. API keys for the live update go in `.env` (template in
`.env.example`).

## What was tested

The hypotheses notebook records the honest result of every idea tried:

- **Squad strength on top of Elo** lowers log-loss in both backtests. *Kept.*
- **Recent form** and **continental-tournament form** barely correlate with
  World Cup performance once Elo is present. *Dropped.*
- A **confederation (CAF) flag** helps one World Cup and hurts the other. *Noise.*
- **Confirmed squads** pull deep nations (Spain, Brazil) below their proxy and
  change the favourites. *Adopted.*
- **Isotonic calibration** can drive the draw probability to zero and explode the
  log-loss; **sigmoid** stays stable. *Sigmoid kept.*
- A **strong defence** predicts deep runs only because good teams have good
  defenders; defensive tilt net of overall quality shows no edge. *No defence bias.*

## Conventions

Project guidelines live in `CLAUDE.md` and `AGENTS.md`: temporal splits only,
no leakage, log-loss and Brier reported alongside accuracy, raw data read-only,
and generated artifacts under `_processed_outputs/`.
