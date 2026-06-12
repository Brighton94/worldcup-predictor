# CLAUDE.md

Behavioral and project guidelines for any AI assistant working in this repository. Pair this file with [AGENTS.md](./AGENTS.md), which is the canonical standard for training, exploration, and modeling notebooks. When the two documents agree, follow both. When they conflict, AGENTS.md wins for notebook structure and CLAUDE.md wins for general behavior.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project Context

- **Goal:** predict football (association football / soccer) match outcomes accurately.
- **Primary targets:**
  1. Match result classification: Home win / Draw / Away win (1X2).
  2. Calibrated probability estimates suitable for betting markets (1X2, and where extended, over/under and BTTS).
- **What "accurate" means here:** well-calibrated probabilities, not just top-1 accuracy. A model that gets 55% accuracy with well-calibrated probabilities is more useful than one with 58% accuracy and overconfident probabilities.
- **Time matters.** Every prediction is made *before kickoff*. Any feature, statistic, or signal that would not be available at kickoff is a leak.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

Football-specific things to clarify before coding:
- Which league(s), seasons, and competitions are in scope.
- Whether matches with extra time or penalties should use 90-minute or full-time results.
- Whether the task wants point predictions, probabilities, or both.
- The time cutoff for any feature used in training (snapshot date, rolling window length).

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Modeling-specific corollaries:
- Always start with a baseline. Reasonable baselines for this project: predict home-win every match, predict the bookmaker-implied probabilities, simple logistic regression on a handful of form features. A new model is only interesting if it beats these on log-loss and Brier score on held-out future matches.
- Prefer one well-understood model with good features over a stack of models with weak features.
- Do not add hyperparameter search, ensembling, or stacking until a single tuned model has been evaluated end-to-end.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

For notebooks, this means: do not re-order cells, re-format markdown, or "tidy" outputs in unrelated sections of a notebook while making a targeted change.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add a model" → "Train on seasons N-1 and earlier, evaluate on season N. Report log-loss, Brier score, and accuracy vs. the home-win baseline and the bookmaker-implied baseline."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after, and that model metrics on a fixed evaluation split are unchanged within tolerance."

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Football Prediction Pitfalls

These are mistakes that look fine in code review but quietly invalidate results. Treat each as a hard rule.

### Temporal splits, never random splits

- Split data by time, not by random shuffle. Train on past, validate on a more recent past, test on future.
- For cross-validation, use forward-chaining / expanding-window CV (`TimeSeriesSplit` or a manual season-based loop). Never `KFold` with `shuffle=True` on match data.
- Within a single match, both rows (home and away perspectives, if duplicated) must land in the same split.

### Leakage to avoid

A feature leaks if it could not have been computed strictly before kickoff. Examples of leakage to refuse:
- Final-score-derived features (xG totals, possession %, shots on target) for the match being predicted.
- Post-match Elo / rating updates that incorporate the match being predicted.
- Season-wide aggregates computed over the full season when predicting a mid-season match.
- League table position, goal difference, or points totals that include the match being predicted.
- Bookmaker closing odds (acceptable as a feature only if the use case is post-hoc analysis; for live prediction use opening or pre-match-snapshot odds and document the snapshot time).

When in doubt, ask: "Would this value have existed on a server 60 seconds before kickoff?"

### Target definition

- The default target is full-time result (90 minutes plus stoppage). Extra time and penalties are excluded unless the task explicitly says otherwise.
- Walkovers, abandonments, and awarded results should be filtered out, not imputed.
- The class encoding convention used in this repo: `H = 0`, `D = 1`, `A = 2` (confirm in code before assuming). Whatever the convention, spell it out in notebook markdown — never rely on bare `0/1/2`.

### Class balance

- Draws are a minority class (typically ~22–28%). A naive classifier will under-predict draws.
- Do not "fix" this with random oversampling or SMOTE without first checking it against a calibration metric — resampling distorts probabilities.
- Prefer class weighting, focal loss, or post-hoc calibration over resampling.

### Calibration

- Report Brier score and log-loss alongside accuracy. For three-class problems, multi-class log-loss is the primary metric.
- Inspect a reliability diagram (calibration curve) for each class before declaring a model "good."
- If the model is miscalibrated, apply isotonic or Platt scaling *fit on a separate calibration fold*, never on the test set.

### Baselines to beat

A new model must be compared against at least these two baselines on the same held-out future matches:
1. **Home-win-always** (or class-prior) — a sanity floor.
2. **Bookmaker-implied probabilities** with the overround removed (proportional normalization, or Shin's method if specified). Beating the bookmaker on log-loss is hard; not beating it is informative.

## Data and Artifact Handling

- Raw data is read-only. Never overwrite source files in cleaning, preprocessing, or feature engineering.
- Generated artifacts (processed datasets, model files, evaluation tables, plots) go under `_processed_outputs/`, `outputs/`, or `artifacts/`, matching the existing convention in this repo. Check before creating a new directory.
- Do not commit raw datasets, scraped CSVs, API responses, secrets, or large model binaries unless explicitly requested.
- Use repo-relative paths or `pathlib.Path`. No hardcoded absolute paths.
- Random seeds are mandatory for any stochastic step. Use a single repo-wide `RANDOM_STATE` constant where possible.

### External APIs and Secrets

- When integrating a new external API (data source, MCP, or third-party service), **always** add a placeholder entry to `.env.example` in the same change set as the code change. The entry must include:
  1. A header comment with the service name and signup URL.
  2. Free-tier limits if known (e.g. "100 req/day, 10 req/min").
  3. A one-line description of what the key unlocks.
  4. Auth header name and base URL when non-obvious.
  5. The `KEY_NAME=your_key_here` line itself.
- Read the corresponding env-var in `src/core/config.py` and surface a warning in `Config.validate()` when it is missing.
- Never commit real keys. `.env` is the local-only secret store; `.env.example` is the public template.

## Modeling Conventions

- Feature matrix is `X`, target is `y`. Splits use `X_train`, `X_test`, `y_train`, `y_test`, plus a `X_val` / `y_val` when a separate calibration or tuning fold is needed.
- Define estimators visibly: `lr = LogisticRegression(...)`, `rf = RandomForestClassifier(random_state=RANDOM_STATE, ...)`. Do not bury estimator construction inside helper functions in teaching notebooks.
- Use `sklearn.pipeline.Pipeline` (or equivalent) when preprocessing must be fit only on training data — particularly for any per-team or per-league aggregation.
- Probability outputs come from `predict_proba`. Store them with explicit names: `proba_home`, `proba_draw`, `proba_away` or a labeled DataFrame.
- For ensembles and stacking, document the base learners, the meta-learner, and the CV scheme used to generate out-of-fold predictions.

## Evaluation Metrics

Report these in this order whenever evaluating a 1X2 model:
1. **Multi-class log-loss** (primary).
2. **Brier score** (multi-class, summed over classes).
3. **Accuracy** and per-class precision / recall / F1 (secondary, for intuition).
4. **Calibration plot** for each class.
5. **Comparison to baselines** (home-win, bookmaker-implied) on the same split.

For betting-market use cases, additionally:
- Expected ROI on a defined staking rule (flat, Kelly, fractional Kelly) computed on out-of-sample matches.
- Closing-line value (CLV) where closing odds are available.

Store evaluation results in a structured DataFrame (e.g. `results_df`) with columns for model name, split, metric, and value. Avoid scattered `print()` statements.

## Notebook Standards

All training, exploration, and analysis notebooks follow [AGENTS.md](./AGENTS.md). In particular:
- Section order, naming conventions, and the pre-modeling flow defined there.
- No emojis in markdown, comments, or printed output.
- No more than two function definitions per cell.
- Restart-and-run-all must work end-to-end before a notebook is considered done.
- Reusable logic moves to `src/` or project modules. Notebooks are for exploration, teaching, and reporting.

## Testing Expectations

- Add or update tests for behavior changes whenever practical.
- Cover edge cases: empty inputs, missing columns, all-NaN groups, matches with no historical fixtures for a team, fixtures outside the configured league set.
- Tests must be deterministic. Seed everything.
- Do not weaken or delete tests to make a change pass.

## Git and Commit Messages

- Single-line commit messages. No multi-paragraph or essay-style commits unless explicitly asked.
- Do not add yourself as a co-author. Commits are authored by the user.
- Do not commit unrelated formatting churn unless the task is specifically formatting.
- Avoid duplicate files with names like `final`, `final_v2`, `copy`, or `latest`.

## Response Format

End substantive responses with a short structured summary:
1. What was changed.
2. What assumptions were made.
3. Which parameters or files to tune first.

---

**These guidelines are working if:** diffs are small and traceable to the request, models are evaluated on temporally-correct splits with calibration reported, baselines are beaten on log-loss before claims of "it works," and clarifying questions arrive before implementation rather than after mistakes.
