# AI Assistant Instructions

These instructions apply to any chatbot, coding assistant, or automation agent generating or modifying code in this repository.

## Project Context

- This repository may contain application code, notebooks, scripts, tests, documentation, and generated artifacts.
- Treat this file as the canonical source of repository-wide AI coding instructions.
- Read the surrounding files, existing patterns, tests, and configuration before making changes.
- Preserve the current project style unless the user explicitly asks for a redesign or refactor.
- Prefer small, targeted changes over broad rewrites.

## Working Style

- Always end your response with a short structured summary:
  1. What was changed
  2. What assumptions were made
  3. Which parameters or files I should tune first

- Match the existing repository structure and coding patterns before introducing new abstractions.
- Preserve the scope of the user request. Do not refactor unrelated files while making a targeted fix.
- Do not overwrite or revert user-authored changes outside the task you are working on.
- Prefer readable, explicit code over clever or overly compact code.
- Explain non-obvious choices in comments or markdown, but avoid excessive commentary.
- Use clear names for variables, functions, files, plots, and outputs.

## Python and Tooling

- Target Python 3.11-compatible code unless the repository clearly specifies another version.
- Use type hints for new or changed function signatures.
- Keep imports at the top of Python files or notebook cells.
- Use `pathlib.Path` or repo-relative paths instead of hardcoded absolute paths.
- Keep code compatible with the configured tooling in files such as `pyproject.toml`, `ruff.toml`, `setup.cfg`, `.pre-commit-config.yaml`, or `tox.ini`.
- Prefer reusable Python logic in `src/`, package modules, or project libraries over notebook-only implementations.
- Add concise docstrings for public functions and non-obvious helpers.
- Do not place imports inside functions unless there is a clear performance or optional-dependency reason.

Before finishing a change, run the relevant checks when possible:

```bash
ruff format .
ruff check .
mypy src
pytest
```

If the project uses different tools, run the closest available equivalents.

## Code Quality Expectations

- Follow PEP 8 and the repository formatter/linter output rather than personal style preferences.
- Write code that is explicit about edge cases, missing data, malformed inputs, empty inputs, and invalid configurations.
- Prefer small, testable functions over large scripts or notebook-only workflows.
- Favor readable names over abbreviations unless the abbreviation is standard in the domain.
- Avoid hidden state and implicit global behavior unless the existing codebase relies on it.
- Keep configuration values in constants, config files, or clearly named variables.
- When working with measurements, units, dates, currencies, IDs, or physical quantities, make assumptions clear in names or comments.
- For stochastic workflows, control randomness with reproducible seeds when repeatability matters.

## Data and Artifact Handling

- Never hardcode machine-specific paths.
- Do not commit raw datasets, secrets, private keys, credentials, tokens, or large generated artifacts unless explicitly requested.
- Keep generated outputs separate from source data by using folders such as `_processed_outputs/`, `_generated/`, `outputs/`, or `artifacts/`.
- Do not overwrite source datasets during cleaning, preprocessing, training, or evaluation.
- Store important intermediate results with clear names when doing so improves reproducibility.
- Avoid modifying existing model artifacts unless the task explicitly involves retraining or replacing them.
- Avoid saving notebook-generated outputs inside the `notebooks/` directory unless the repository already follows that convention.

## Notebook-Specific Rules

- Use clear, professional language in notebook markdown, comments, and printed output.
- Do not use emojis in markdown cells, comments, print output, or status messages.
- Structure notebooks clearly with sections such as Title, Learning Objectives, Outline, Introduction, Imports, Config, Data Loading, EDA, Data Cleaning, Feature Engineering, Modeling, Evaluation, Interpretation, Conclusion, and optional Appendix or References.
- Add brief markdown lead-ins before major code cells so the reader knows what the next cell does, why it is being run, and how to interpret the result.
- Keep cells small and focused, with one task per cell.
- Ensure notebooks work with Restart and Run All before considering them complete.
- Avoid hardcoded paths and magic values. Use constants, configuration variables, and proper file handling.
- Move reusable logic into Python modules and import it into notebooks.
- Use notebooks for exploration, teaching, reporting, and prototyping, not for production system logic.
- Do not define more than two functions in one notebook cell.
- Avoid long runs of sequential `print()` calls. Prefer one formatted message, a table, or a short formatted block.
- Create meaningful visualizations with clear titles, axis labels, legends, and units where relevant.
- Store important results in variables, files, or structured outputs instead of relying only on printed output.
- Manage memory carefully, especially with large datasets, image stacks, embeddings, or model outputs.
- Clean notebook outputs before sharing when appropriate by clearing outputs and rerunning all cells.
- Version notebooks cleanly. Prefer notebook-friendly tooling such as `nbstripout` or Jupytext when appropriate.
- Avoid creating multiple duplicate notebook versions with names like `final`, `final_v2`, `copy`, or `latest`.

## Pre-Modeling Notebook Requirements

Use these rules for notebooks focused on data understanding, cleaning, wrangling, feature preparation, and statistical analysis before machine learning modeling.

### General Flow

- Prefer this section order when appropriate:
  1. Title
  2. Learning Objectives
  3. Outline
  4. Introduction
  5. Imports
  6. Configuration
  7. Data Loading
  8. Initial Inspection
  9. Data Cleaning
  10. Missing Value Handling
  11. Exploratory Data Analysis
  12. Hypothesis Testing or Statistical Checks
  13. Feature Engineering
  14. Final Prepared Dataset
  15. Conclusion

- Keep the teaching progression simple:
  1. inspect the raw data
  2. identify data quality issues
  3. clean and transform the data
  4. explain the reasoning
  5. validate the cleaned result
  6. prepare the data for modeling or reporting

### Data Loading and Initial Inspection

- Load data with explicit, repo-relative paths or clearly named constants.
- Display the first few rows with `df.head()`.
- Inspect shape, columns, data types, missing values, and summary statistics early.
- Use familiar checks such as:
  - `df.shape`
  - `df.info()`
  - `df.describe()`
  - `df.isna().sum()`
  - `df.duplicated().sum()`
  - `df.nunique()`
  - `df.value_counts()` for categorical columns

### Data Cleaning

- Preserve the original raw dataframe where useful, for example `df_raw = df.copy()`.
- Use clear cleaned dataframe names such as `df_clean`, `df_model`, or `df_features`.
- Explain each cleaning decision in markdown before applying it.
- Handle duplicate rows explicitly and report how many were removed.
- Standardize column names when useful, for example lowercase names with underscores.
- Convert data types explicitly for dates, categories, booleans, numeric values, and IDs.
- Validate conversions after applying them.
- Do not silently drop rows or columns. Explain why data is removed.
- When replacing values, document the before-and-after representation.

### Missing Values and NaNs

- Always inspect missing values before filling or dropping them.
- Show missingness with a count table, percentage table, or visualization.
- Prefer explicit missing-value strategies:
  - drop rows only when justified
  - fill numeric values with mean, median, constant, or grouped statistics when appropriate
  - fill categorical values with mode, `"Unknown"`, or a documented category when appropriate
  - use forward fill or backward fill only for ordered/time-series data where it makes sense
- Store imputation choices in clearly named variables where helpful.
- Recheck missing values after cleaning with `df.isna().sum()`.
- Avoid data leakage by fitting imputers only on training data once a train/test split exists.

### Exploratory Data Analysis

- Examine both univariate and multivariate patterns.
- Plot continuous variables with histograms, boxplots, KDE plots, or summary tables.
- Plot categorical variables with count tables and bar charts.
- Use scatter plots, pair plots, correlation matrices, grouped summaries, and pivot tables where useful.
- Make plot titles and axis labels specific and readable.
- For target variables, inspect the distribution before modeling.
- For categorical targets, inspect class balance.
- For continuous targets, inspect skewness, outliers, range, and transformations.
- Use correlation analysis carefully and explain that correlation does not imply causation.

### Outliers

- Detect outliers with clear methods such as boxplots, IQR rules, z-scores, robust statistics, or domain thresholds.
- Do not remove outliers automatically.
- Explain whether outliers are likely errors, rare valid values, or important signals.
- If outliers are capped, transformed, or removed, report the number of affected rows.
- Keep the original value available where auditability matters.

### Hypothesis Testing and Statistical Checks

- State the null hypothesis and alternative hypothesis in markdown before running a test.
- Define the significance level, commonly `alpha = 0.05`, before interpreting results.
- Check assumptions where appropriate, such as normality, equal variance, independence, or sample size.
- Use appropriate tests for the question, for example:
  - t-test for comparing two means
  - ANOVA for comparing more than two means
  - chi-square test for categorical relationships
  - correlation tests for numeric relationships
  - non-parametric tests when parametric assumptions are not reasonable
- Report the test statistic, p-value, and decision.
- Explain the result in plain language, not only as "reject" or "fail to reject."
- Avoid claiming practical importance from statistical significance alone. Include effect sizes or descriptive differences where helpful.

### Feature Engineering

- Create features with clear names and documented reasoning.
- Keep feature engineering steps visible in notebooks before moving reusable logic into modules.
- Handle dates by extracting useful components such as year, month, day, weekday, elapsed time, or age where relevant.
- Encode categorical variables explicitly with a clear method such as one-hot encoding, ordinal encoding, or mapping.
- Scale or transform numeric variables when needed, and explain why.
- Use log, square-root, binning, interaction, ratio, or polynomial features only when they are meaningful.
- Avoid target leakage. Do not create features using information that would not be available at prediction time.
- After feature engineering, display the updated dataframe and verify the new columns.
- Keep a final feature list in a clearly named variable such as `feature_cols`.

### Handoff to Modeling

- End pre-modeling notebooks with a clear prepared dataset or feature matrix.
- Summarize what was cleaned, transformed, removed, or engineered.
- Save processed data only to an output directory, not over the raw input.
- Make the next modeling step obvious by defining useful variables such as `target_col`, `feature_cols`, `X`, and `y` where appropriate.

## Classification Notebook Requirements

Use these rules for notebooks that teach, demonstrate, or explain classification workflows.

- Define the feature matrix as `X` and the target as `y`.
- Use standard split variables: `X_train`, `X_test`, `y_train`, `y_test`.
- Define the target and feature matrix explicitly in code and explain them in markdown before fitting models.
- Inspect target balance before training with a count table, bar chart, or both.
- Define estimators visibly in the notebook with readable variable names, such as `lm = LogisticRegression()`.
- Fit models with direct calls such as `lm.fit(X_train, y_train)`.
- Use explicit prediction variables such as `y_pred`, `pred_lm`, or `y_pred_proba`.
- Do not hide the primary classification workflow entirely inside helper functions when the notebook is meant to teach the modeling process.
- Keep the feature definition, split, model definition, fit, prediction, and evaluation steps visible.
- When scaling is needed, show the scaling step explicitly with familiar variables such as `X_scaled = scaler.fit_transform(X)` or with a train/test-safe equivalent:
  - `scaler.fit(X_train)`
  - `X_train_scaled = scaler.transform(X_train)`
  - `X_test_scaled = scaler.transform(X_test)`
- Explain common classification metrics in plain language:
  - accuracy
  - precision
  - recall
  - F1-score
  - ROC-AUC when appropriate
  - confusion matrix
- Spell out class meanings explicitly instead of relying on shorthand labels such as `0`, `1`, `True 0`, or `Pred 1` without explanation.
- Include meaningful evaluation outputs such as `classification_report`, confusion matrices, ROC curves, precision-recall curves, or cross-validation scores where useful.
- Prefer a teaching progression:
  1. baseline example
  2. interpretation
  3. tuning or improved variant
  4. comparison of results

## Regression Notebook Requirements

Use these rules for notebooks that teach, demonstrate, or explain regression workflows.

- Define the feature matrix as `X` and the continuous target as `y`.
- Use standard split variables: `X_train`, `X_test`, `y_train`, `y_test`.
- Define the target and feature matrix explicitly in code and explain them in markdown before fitting models.
- Inspect the target distribution before modeling using summary statistics and visualizations.
- For continuous targets, check range, skewness, outliers, and possible transformations.
- Define estimators visibly in the notebook with readable names, such as:
  - `lm = LinearRegression()`
  - `ridge = Ridge(alpha=1.0)`
  - `lasso = Lasso(alpha=0.1)`
  - `tree = DecisionTreeRegressor(random_state=RANDOM_STATE)`
  - `rf = RandomForestRegressor(random_state=RANDOM_STATE)`
- Fit models with direct calls such as `lm.fit(X_train, y_train)`.
- Use explicit prediction variables such as `y_pred`, `pred_lm`, or `y_test_pred`.
- Evaluate regression models with clear metrics such as:
  - MAE
  - MSE
  - RMSE
  - R²
  - cross-validation scores where appropriate
- Plot actual vs predicted values where useful.
- Plot residuals and explain whether they show bias, non-linearity, heteroscedasticity, or unusual errors.
- For linear models, inspect coefficients where interpretation matters.
- For regularized models, explain the role of `alpha` and compare Ridge and LASSO behavior where useful.
- When scaling is needed, show the scaling step explicitly and avoid leakage by fitting scalers on training data only after a split.
- For tree-based regression, show feature importance where appropriate and explain limitations.
- For model comparison notebooks, store metrics in a dataframe such as `results_df` and compare models clearly.
- Prefer a teaching progression:
  1. simple baseline
  2. model interpretation
  3. regularized or tuned variant
  4. comparison and conclusion

## Model Development and Evaluation

- Start with a simple baseline before introducing complex models.
- Keep model inputs and outputs clear.
- Separate training, validation, testing, and inference logic where possible.
- Avoid data leakage from preprocessing, feature engineering, scaling, imputation, or target-derived features.
- Use pipelines when they improve reproducibility and reduce leakage.
- Use cross-validation when appropriate, especially for small datasets.
- Store model metrics in structured variables or tables.
- Explain important trade-offs, assumptions, and limitations.
- Save trained models only when explicitly needed, and include the preprocessing steps required to use them correctly.

## Testing Expectations

- Add or update tests for behavior changes whenever practical.
- Use descriptive test names that reflect the expected behavior.
- Cover edge cases such as empty inputs, malformed records, missing files, missing columns, invalid values, and no-result cases.
- Prefer named constants in tests when they improve clarity and avoid unexplained magic values.
- Keep tests deterministic by controlling randomness and avoiding reliance on external services unless mocked.
- Do not weaken or delete tests simply to make a change pass.

## Documentation Expectations

- Update README files, docstrings, examples, or notebooks when behavior changes.
- Keep documentation accurate and close to the code it describes.
- Include usage examples for public APIs or user-facing scripts when useful.
- Document assumptions, configuration options, inputs, outputs, and expected file formats.
- Avoid over-documenting obvious code.

## Git and Commit Messages

- Write single-line commit messages.
- Do not generate essay-style or multi-paragraph commit messages unless explicitly asked.
- Avoid duplicate generated files with names like `final`, `copy`, or `new`.
- Do not commit unrelated formatting churn unless the task is specifically formatting.

## Preferred Contribution Pattern

1. Read the surrounding module, notebooks, tests, and repo configuration before changing code.
2. Identify the smallest change that solves the task cleanly.
3. Implement the change using existing project conventions.
4. Add or update tests and documentation where behavior changes.
5. Run the relevant checks when possible.
6. Summarize assumptions, limitations, and follow-up work clearly.
