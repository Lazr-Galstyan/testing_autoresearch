# autoresearch: churn, approach 3 (raw line items)

An autonomous-research experiment. You (the agent) improve a churn model by editing
ONE file, `autoresearch_raw_data/train.py`, and keeping only changes that raise the
cross-validated AUC-ROC. You run exactly 100 experiments, then stop.

## The task

Predict `churn` for each customer from raw transaction line items in
`data/processed/raw_train.csv`. There is one row per item on a customer's FIRST
invoice, with these columns and nothing derived:

`invoice, stock_code, description, quantity, invoice_date, price, customer_id, country, churn`

`churn` = 1 if the customer makes no second purchase within 90 days of this first
invoice, else 0 (the label is the same on every line of a customer). You build every
feature yourself from these line items: there are no pre-built features, no embeddings
and no customer table. 4,035 customers, 99,830 lines. Higher AUC-ROC is better.

## Setup

1. Agree a run tag with the human. Branch `autoresearch_raw_data/<tag>` must not exist.
2. `git checkout -b autoresearch_raw_data/<tag>` from the current HEAD.
3. Read `autoresearch_raw_data/prepare.py` (fixed harness) and `autoresearch_raw_data/train.py` (yours).
4. Verify `data/processed/raw_train.csv` exists; if not, tell the human to run `python autoresearch_raw_data/build_data.py`.
5. Create `autoresearch_raw_data/results.tsv` with only the header row (all seven columns, see Logging).
6. Run the unmodified `train.py` first to record the baseline (experiment 1), copy it to
   `autoresearch_raw_data/best_train.py`, then start the loop.

## The interface

```python
def make_features(lines_fit, y_fit, lines_apply):  # -> (X_fit, X_apply)
def make_model():                                   # -> xgboost.XGBClassifier
```

- `lines_fit`, `lines_apply`: line-item DataFrames with the columns above except `churn`.
- `y_fit`: churn label of the fit customers, a Series indexed by `customer_id`.
- Return one feature row per customer, indexed by `customer_id`, for both. The harness
  checks this and that the columns match. Customers with no usable data still need a row.
- Anything learned from data (category levels, scalers, target or count encodings,
  selected columns) must be computed from `lines_fit` / `y_fit` only and applied to
  `lines_apply`. The harness calls this once per fold with that fold's earlier
  customers as fit and its later customers as apply. Never use `lines_apply` to learn
  anything.

## What you CAN do

- Edit `autoresearch_raw_data/train.py` only: feature engineering from the line items
  (aggregations, time features from `invoice_date`, basket composition, price
  statistics, text features from `description`, country handling, etc.), feature
  selection, and the XGBoost hyperparameters, early stopping, class weights.
- Search the web and read docs/Kaggle write-ups. Do this before your first
  non-baseline experiment, every 10 experiments, and after 3 consecutive discards. Cite
  the source in your hypothesis.
- Use only packages already in `requirements.txt`. Do not install anything.

## What you CANNOT do

- Use any predictor other than XGBoost (`xgboost.XGBClassifier`). The model returned by
  `make_model` must be XGBoost.
- Edit `prepare.py`, the folds, or the metric. `run_cv` is the ground truth.
- Read, load or reference anything else under `data/` (only `raw_train.csv` is yours),
  `data/processed/raw_test.csv`, `results/`, `autoresearch_raw_data/final_eval.py`,
  `autoresearch_raw_data/build_data.py`, the `autoresearch/` folder (another run's
  results), `src/` (the hand-built feature code and the conventional tuning),
  `conventional_approach/` and `notebooks/`. The test set, the engineered features and
  other approaches' results are off limits; using them would invalidate the comparison.
- Use git history or any file to peek at earlier scores other than your own `results.tsv`.
- Use `customer_id` as a feature (it is an identifier, not a signal).
- Run commands that read code from stdin (heredocs, `python3 -`) or `source` files
  outside the project. Put helper scripts inside `autoresearch_raw_data/` and run them
  as `python3 autoresearch_raw_data/<file>.py`; use the Edit tool for log files.

## Goal and simplicity

Maximise `CV AUC` (mean of 4 time-ordered folds). Simpler is better: a +0.001 gain that
adds 20 hacky lines is not worth keeping; equal AUC with less code is a win. The signal
is weak and fold AUCs vary by about 0.01, so treat gains smaller than ~0.002 as likely
noise and prefer changes that help most folds.

## Output

`python3 autoresearch_raw_data/train.py > autoresearch_raw_data/run.log 2>&1` prints:

```
Fold AUCs: [...]
CV time: 0.2s
CV AUC: 0.5947
```

Read the metric with `grep "^CV AUC:" autoresearch_raw_data/run.log`. Empty grep = crash;
read `tail -n 50`. Each run should take well under a minute; kill and discard anything
over 2 minutes.

## Logging

`autoresearch_raw_data/results.tsv`, tab-separated, header + one row per experiment:

```
experiment	model	variables	modification	CV_AUC	cv_seconds	status
```

- `experiment`: 1 to 100.
- `model`: the XGBoost model and its hyperparameters as run.
- `variables`: the COMPLETE list of the features (the columns of the customer table
  returned by `make_features`) that the model used in this experiment, comma-separated
  and written out by name, not summarised (e.g. `n_lines, total_spend, recency_days, ...`).
  List every feature of this experiment, not only the new ones, and fill it in for
  every row, including the baseline, discards and crashes (for a crash, list what the
  code was meant to create). Write it from the code you actually ran.
- `modification`: what you changed relative to the best version so far (feature
  engineering, parameters, etc.). For the baseline, write `baseline`.
- `CV_AUC`: the mean CV AUC (0.0000 for crashes).
- `cv_seconds`: wall-clock time of the CV run.
- `status`: `keep`, `discard` or `crash`.

Also keep `autoresearch_raw_data/research-log.md`: per experiment, the hypothesis, whether
it was a follow-up / ablation / exploration, and what you observed. Keep every row,
discarded and crashed ones included. These files stay in git for the human; do not use
git commits to record anything yourself: **never `git commit`**.

## The loop

Run **exactly 100 experiments in total**, baseline included, then stop. Crashed and
discarded runs count towards the 100. Do not ask whether to continue before then.

Nothing is committed. To be able to undo a bad change, keep
`autoresearch_raw_data/best_train.py`, a copy of the best `train.py` so far.

1. Review `results.tsv`. State a hypothesis; avoid near-duplicate experiments.
2. Edit `train.py`.
3. Run it and read the CV AUC.
4. Higher than the best so far: `keep`; copy `train.py` over `best_train.py`.
   Equal or lower: `discard`; copy `best_train.py` back over `train.py`.
5. Append the row to `results.tsv` (with the model and the complete variable list) and the entry to `research-log.md`.
6. Every 10 experiments, pause to synthesise what works and what does not, then continue.

After experiment 100, make sure `train.py` holds the best version, then stop and report:
best CV AUC, the kept changes, and the 100 results. The human runs the held-out test
evaluation, not you.
