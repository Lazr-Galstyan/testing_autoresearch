# autoresearch: churn, shared features, XGBoost only

An autonomous-research experiment. You (the agent) conduct research in optimizing an
XGBoost model for customer churn by editing ONE file, `autoresearch_xgboost_only/train.py`,
and keeping only changes that raise the cross-validated AUC-ROC. You run exactly 100
experiments, then stop.

## The task

Predict `churn` for each customer from `data/processed/train.csv` (4,035 customers,
173 features: aggregates, one-hot columns, `pca_*` columns, date parts). Higher AUC-ROC
is better. The starting point is a plain, untuned XGBoost with no feature engineering.

## Setup

1. Agree a run tag with the human. Branch `autoresearch_xgboost_only/<tag>` must not exist.
2. `git checkout -b autoresearch_xgboost_only/<tag>`. Do this directly: do NOT switch
   branches first. Branch from whatever HEAD is currently at.
3. Read the in-scope files. The folder is small:
   - `autoresearch_xgboost_only/prepare.py`: the fixed harness (data, folds, metric). Do not modify.
   - `autoresearch_xgboost_only/train.py`: the file you modify. Data preparation, feature
     engineering, choosing hyperparameters and model training (with possible early stopping etc.).
4. Verify `data/processed/train.csv` exists. If not, tell the human to run `python src/split_data.py`.
5. Create `autoresearch_xgboost_only/results.tsv` with only the header row (all seven columns, see Logging).
6. Run the unmodified `train.py` first to establish the baseline (experiment 1), copy it to
   `autoresearch_xgboost_only/best_train.py`, then start the loop.
7. Confirm setup looks good and go.

## Experimentation

You launch an experiment with `python3 autoresearch_xgboost_only/train.py`.

**What you CAN do:**
- Modify `train.py`: this is the only file you edit. Everything is fair game that will lead
  to a model that generalizes to later customers: data preparation, feature engineering,
  choosing hyperparameters, and model training. You can also implement new features such
  as early stopping etc.
- Search the web and read external resources. This is not optional: you MUST do research
  before relying solely on your own intuition. See the **Research** section below.

**What you CANNOT do:**
- Use any predictor other than XGBoost. The model returned by `make_model` must be
  XGBoost (`xgboost.XGBClassifier`) or a combination of several XGBoost models (for
  example different seeds, depths or feature subsets, averaged or blended). No other
  model family may be used as a predictor or to generate features (no logistic regression,
  random forests, kNN, neural networks, etc.). Non-predictive transformations such as
  encodings, scaling, binning or PCA are feature engineering and are allowed.
- Modify `prepare.py`, the folds, or the metric. `run_cv` is the ground truth.
- Install new packages or add dependencies. You can only use what is already installed.
- Read, load, or reference `data/processed/test.csv`, anything in `results/`,
  `autoresearch_xgboost_only/final_eval.py`, `src/xgboost_tuning.py`, `conventional_approach/`,
  the `autoresearch/` and `autoresearch_raw_data/` folders (other approaches' results and
  code), or any other data file than `data/processed/train.csv`. The test set and the other
  approaches' answers are off limits; using them would invalidate the comparison.
- Use git to peek at earlier results, or any file with earlier scores other than your own
  `results.tsv`.
- Use `first_date` or anything that identifies a customer. The harness already removes them.
- Fit anything on validation rows. All learned state (scalers, category levels,
  target/count encodings, selected columns) must be computed from `X_fit` inside
  `make_features` and applied to `X_apply`. The folds are time-ordered: later customers
  are validated using only earlier ones, mirroring the real test split.
- Run commands that read code from stdin (heredocs, `python3 -`) or `source` files outside
  the project. Put helper scripts inside `autoresearch_xgboost_only/` and run them as
  `python3 autoresearch_xgboost_only/<file>.py`; use the Edit tool for log files.

## Interface

```python
def make_features(X_fit, X_apply):  # -> (X_fit', X_apply')
def make_model():                   # -> estimator with fit(X, y) and predict_proba(X), XGBoost only
```

`make_features` is called once per fold. `X_fit` holds that fold's earlier customers and
`X_apply` its later ones. Both are DataFrames without the label. Return both with the same
columns. Keep ALL feature engineering inside `make_features`, because the final
evaluation by the human runs exactly this function.

## Research

You are expected to actively search the web and read external resources throughout the
experiment. Do not treat this as a fallback for when you are stuck: treat it as a primary
input to your experiment design.

**When to research:**
- Before your very first non-baseline experiment: search for best practices on tuning
  XGBoost for binary classification, common feature engineering techniques for tabular
  data, and known good hyperparameter ranges.
- Every 10 experiments (aligned with your synthesis pause): search for new ideas you have
  not tried yet. Look for Kaggle write-ups, blog posts, papers, or documentation pages
  relevant to what you are working on.
- Whenever you hit a plateau (3+ consecutive discards with <0.001 movement): stop and
  research before trying another random tweak. Look for techniques specifically designed to
  break through plateaus: different feature transformations, interaction terms, encoding
  strategies, or XGBoost parameters you have not explored.
- When trying a new category of change (e.g. first time doing feature engineering, first
  time tuning categorical handling): read about that specific topic first.

**What to search for:**
- XGBoost documentation for parameters you have not tried
- "XGBoost tuning guide" / "XGBoost best practices"
- "feature engineering for customer churn prediction" or similar domain-specific resources
- Kaggle notebooks and competition solutions for similar tabular classification problems
- Academic papers or blog posts on gradient boosting optimization

**How to use what you find:**
- Cite your source when proposing an experiment (e.g. "Based on [XGBoost docs on
  regularization], trying...").
- Do not blindly copy: adapt what you read to this specific dataset and problem.
- If a source suggests a technique, understand *why* it works before implementing it.

Research time does not count against the experiment timeout.

## Goal and simplicity

**The goal is simple: get the highest CV AUC.** Everything is fair game that will lead to a
model that generalizes to later customers: data preparation, feature engineering, choosing
hyperparameters, and model training. Be creative!

**Simplicity criterion**: all else being equal, simpler is better. A small improvement that
adds ugly complexity is not worth it. Conversely, removing something and getting equal or
better results is a great outcome: a simplification win. When evaluating whether to keep a
change, weigh the complexity cost against the improvement magnitude. A 0.001 AUC improvement
that adds 20 lines of hacky code? Probably not worth it. A 0.001 AUC improvement from
deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

The signal in this data is weak and fold AUCs vary by about 0.01, so treat gains smaller
than ~0.002 as likely noise and prefer changes that help most folds.

## Output format

`python3 autoresearch_xgboost_only/train.py > autoresearch_xgboost_only/run.log 2>&1` prints:

```
Fold AUCs: [...]
CV time: 1.9s
CV AUC: 0.5712
```

Read the key metric with `grep "^CV AUC:" autoresearch_xgboost_only/run.log`. An empty grep
means the run crashed: read `tail -n 50`. Each run should take well under a minute; kill
and discard anything over 2 minutes.

## Logging results

When an experiment is done, log it to `autoresearch_xgboost_only/results.tsv` (tab-separated,
NOT comma-separated: commas break in descriptions). The TSV has a header row and 7 columns:

```
experiment	model	variables	modification	CV_AUC	cv_seconds	status
```

- `experiment`: 1 to 100.
- `model`: the XGBoost model(s) and hyperparameters as run.
- `variables`: the COMPLETE list of the features (the columns of the tables returned by
  `make_features`) that the model used in this experiment, comma-separated and written out
  by name, not summarised. List every feature of this experiment, not only the new ones, and
  fill it in for every row, including the baseline, discards and crashes (for a crash, list
  what the code was meant to create). Write it from the code you actually ran.
- `modification`: what you changed relative to the best version so far. For the baseline,
  write `baseline`.
- `CV_AUC`: the mean CV AUC (0.0000 for crashes).
- `cv_seconds`: wall-clock time of the CV run.
- `status`: `keep`, `discard` or `crash`.

## Research log

Also maintain a research log, `autoresearch_xgboost_only/research-log.md`, with details of
your thinking, hypotheses, and observations for each experiment. This helps track your
reasoning and decisions over time. Make it so that it can be related to `results.tsv` by
experiment number. Keep every row, discarded and crashed ones included. These files stay in
git for the human; do not use git commits to record anything yourself: **never `git commit`**.

## The experiment loop

Run **exactly 100 experiments in total**, baseline included, then stop. Crashed and
discarded runs count towards the 100. Do not ask whether to continue before then.

Nothing is committed. To be able to undo a bad change, keep
`autoresearch_xgboost_only/best_train.py`, a copy of the best `train.py` so far.

1. Look at `results.tsv`.
2. **Choose your next experiment deliberately.** Before touching any code:
   - Review `results.tsv` and your research log.
   - State a short **hypothesis**: what you are changing, why you think it will help, and
     (if applicable) which prior result motivates this step.
   - Classify the experiment as one of: *follow-up* to a promising result,
     *ablation/simplification* of a promising result, or *exploration* of a meaningfully
     different direction.
   - **Do not** run near-duplicate experiments unless you can state exactly what is
     different and why it matters. Avoid random-walk behavior and cosmetic variations of
     the same idea.
   - If you have not done web research in the last 10 experiments, or if your last 3+
     experiments were discards, do research now before proposing your next change.
3. Tune `train.py` with that experimental idea by directly hacking the code.
4. Run the experiment and read the CV AUC (see **Output format**). If the grep output is
   empty, the run crashed: read the stack trace and attempt a fix. If you cannot get things
   to work after more than a few attempts, give up on the idea and log a `crash`.
5. Higher than the best so far: `keep`; copy `train.py` over `best_train.py`.
   Equal or lower: `discard`; copy `best_train.py` back over `train.py`.
6. Append the row to `results.tsv` (with the model and the complete variable list) and the
   entry to the research log.
7. **Every 10 experiments**, pause and briefly synthesise what you have learned so far: what
   kinds of changes help, what kinds do not, what your current best theory is about what
   matters on this dataset, and what direction to try next. Write this synthesis as a short
   note in your context (not a file) to inform subsequent experiments.

The idea is that you are a completely autonomous researcher trying things out. If they work,
keep. If they do not, discard. If you feel like you are getting stuck, think harder: read
the sources you found, re-read your log for new angles, try combining previous near-misses.

**Crashes**: if a run crashes (OOM, a bug, etc.), use your judgment. If it is something
dumb and easy to fix (a typo, a missing import), fix it and re-run. If the idea itself is
fundamentally broken, skip it, log `crash`, and move on.

After experiment 100, make sure `train.py` holds the best version, then stop and report:
best CV AUC, the kept changes, and the 100 results. The human runs the held-out test
evaluation, not you.
