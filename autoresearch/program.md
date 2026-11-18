# autoresearch: churn, approach 1 (shared features)

An autonomous-research experiment. You (the agent) improve a churn model by
editing ONE file, `autoresearch/train.py`, and keeping only changes that raise the
cross-validated AUC-ROC. You run exactly 100 experiments, then stop.

## The task

Predict `churn` for each customer from `data/processed/train.csv` (4,035 customers,
173 features: aggregates, one-hot columns, `pca_*` columns, date parts). Higher
AUC-ROC is better. A conventional tuned XGBoost reaches CV AUC ~0.608 on the same
folds; the untuned starting point scores ~0.571.

## Setup

1. Agree a run tag with the human. Branch `autoresearch/<tag>` must not exist.
2. `git checkout -b autoresearch/<tag>` from the current HEAD.
3. Read: `autoresearch/prepare.py` (fixed harness), `autoresearch/train.py` (yours).
4. Verify `data/processed/train.csv` exists; if not, tell the human to run `python src/split_data.py`.
5. Create `autoresearch/results.tsv` with only the header row (all seven columns, see Logging).
6. Run the unmodified `train.py` first to record the baseline (this is experiment 1), then start the loop.

## What you CAN do

- Edit `autoresearch/train.py` only. Feature engineering, encodings, feature
  selection, model choice, hyperparameters, early stopping, class weights: all fair
  game, provided it generalises to later customers.
- Search the web and read docs/Kaggle write-ups. Do this before your first
  non-baseline experiment, every 10 experiments, and after 3 consecutive discards.
  Cite the source in your hypothesis.
- Use only packages already in `requirements.txt` (xgboost, scikit-learn, pandas,
  numpy, ...). Do not install anything.

## What you CANNOT do

- Edit `prepare.py`, the folds, or the metric. `run_cv` is the ground truth.
- Read, load, or reference `data/processed/test.csv`, `results/`, `autoresearch/final_eval.py`,
  `src/xgboost_tuning.py`, or `conventional_approach/`. The test set and the
  conventional answer are off limits; looking at them would invalidate the comparison.
- Use git history or any file to peek at earlier scores other than your own `results.tsv`.
- Use `first_date` or anything that identifies a customer. The harness already removes them.
- Fit anything on validation rows. All learned state (scalers, category levels,
  target/count encodings, selected columns) must be computed from `X_fit` inside
  `make_features` and applied to `X_apply`. The folds are time-ordered: later
  customers are validated using only earlier ones, mirroring the real test split.

## Goal and simplicity

Maximise `CV AUC` (mean of 4 time-ordered folds). Simpler is better: a +0.001 gain that
adds 20 hacky lines is not worth keeping; equal AUC with less code is a win. The signal
in this data is weak, so treat gains smaller than ~0.002 as likely noise (fold AUCs
vary by ~0.01); prefer changes that help most folds.

## Output

`python3 autoresearch/train.py > autoresearch/run.log 2>&1` prints:

```
Fold AUCs: [...]
CV time: 1.9s
CV AUC: 0.5712
```

Read the metric with `grep "^CV AUC:" autoresearch/run.log`. Empty grep = crash; read
`tail -n 50`. Each run should take well under a minute; kill and discard anything over 2 minutes.

## Logging

`autoresearch/results.tsv`, tab-separated, header + one row per experiment:

```
experiment	model	variables	modification	CV_AUC	cv_seconds	status
```

- `experiment`: 1 to 100.
- `model`: the model and its hyperparameters as run, e.g.
  `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, ...)`.
- `variables`: the COMPLETE list of the features (the columns of the tables returned by
  `make_features`) that the model used in this experiment, comma-separated and written out
  by name, not summarised. List every feature of this experiment, not only the new ones, and
  fill it in for every row, including the baseline, discards and crashes (for a crash, list
  what the code was meant to create). Write it from the code you actually ran.
- `modification`: what you changed in this experiment relative to the best version so
  far (feature engineering, parameters, etc.). For the baseline, write `baseline`.
- `CV_AUC`: the mean CV AUC (0.0000 for crashes).
- `cv_seconds`: wall-clock time of the CV run.
- `status`: `keep`, `discard` or `crash`.

Also keep `autoresearch/research-log.md`: per experiment, the hypothesis, whether it was a
follow-up / ablation / exploration, and what you observed. Keep every row, discarded and
crashed ones included. Do not use git commits to record anything: **never `git commit`**.

## The loop

Run **exactly 100 experiments in total**, baseline included, then stop. Crashed and
discarded runs count towards the 100. Do not ask whether to continue before then.

Nothing is committed. To be able to undo a bad change, keep `autoresearch/best_train.py`,
a copy of the best `train.py` so far (created after the baseline).

1. Review `results.tsv`. State a hypothesis; avoid near-duplicate experiments.
2. Edit `train.py`.
3. Run it and read the CV AUC.
4. Higher than the best so far: `keep`; copy `train.py` over `best_train.py`.
   Equal or lower: `discard`; copy `best_train.py` back over `train.py`.
5. Append the row to `results.tsv` (with the model and the complete variable list) and the entry to `research-log.md`.
6. Every 10 experiments, pause to synthesise what works and what does not, then continue.

After experiment 100, make sure `train.py` holds the best version, then stop and
report: best CV AUC, the kept changes, and the 100 results. The human runs the
held-out test evaluation, not you.
