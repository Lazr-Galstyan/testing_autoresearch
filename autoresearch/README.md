# Autoresearch

An LLM coding agent improves a churn model on its own: it edits one file, scores the change with a fixed cross-validation harness, keeps the change if the score went up, discards it otherwise, and repeats. This directory holds **approach 2 of 3** in the project (see the root `README.md`): autoresearch on the **same processed data** the conventional XGBoost pipeline uses.

The method comes from two sources:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch): the fixed-harness / one-editable-file split, the keep-or-discard loop, `results.tsv` and the simplicity criterion.
- [szilard/xgboost-autoresearch](https://github.com/szilard/xgboost-autoresearch): the adaptation to XGBoost, where the agent does feature engineering and tuning, states a hypothesis per experiment, researches on the web, and is checked afterwards by a human-only ground-truth script.

## What it does

The agent starts from a plain, untuned XGBoost on `data/processed/train.csv` (173 features) and tries to maximise the **mean validation AUC-ROC over four time-ordered, expanding-window folds** (training on the earliest 60 / 70 / 80 / 90 % of customers). Those folds are the same ones the conventional pipeline tunes against, so the two scores are comparable. The agent may change feature engineering, encodings, the model and its hyperparameters. It may not touch the evaluation, the test set, or the conventional results.

When the experiment budget is spent, a human runs `final_eval.py` once. It fits the agent's final pipeline on the whole training set and scores it on the held-out test set.

## Files

| File | Edited by | Purpose |
|---|---|---|
| `prepare.py` | nobody (fixed) | Loads `train.csv` only, builds the time-ordered folds, and `run_cv` scores the agent's pipeline. Never loads the test set. |
| `train.py` | the agent | `make_features(X_fit, X_apply)` and `make_model()`; running it prints the fold AUCs and the CV AUC. |
| `program.md` | the human | The agent's instructions: setup, allowed and forbidden actions, the loop, logging. |
| `final_eval.py` | nobody (human-only) | Fits on all of train, evaluates once on test, writes `results/autoresearch_shared_features.json`. The agent must not read or run it. |
| `results.tsv`, `research-log.md` | the agent | Per-experiment log and reasoning log. Tracked in git. |
| `best_train.py`, `run.log` | the agent | Backup of the best `train.py` and the latest run output. Git-ignored. |

## How an experiment runs

1. The agent states a hypothesis and edits `train.py`.
2. `python3 autoresearch/train.py` runs the fixed CV (about 2 s).
3. Higher CV AUC than the best so far: the change is kept (and copied to `best_train.py`). Otherwise `train.py` is restored from `best_train.py`. Nothing is committed.
4. The result (experiment number, model, modification, CV AUC, time, status) goes in `results.tsv` and the reasoning in `research-log.md`. The loop runs exactly 100 experiments, baseline included.

`make_features` is called once per fold with that fold's training rows as `X_fit` and its validation rows as `X_apply`. Anything learned from data (category levels, scalers, target encodings) must come from `X_fit` only, so nothing from a later period leaks into an earlier one.

## Running it

```bash
python src/split_data.py                      # once, if data/processed/train.csv is missing
python3 autoresearch/train.py                 # sanity check: prints the baseline CV AUC
git checkout -b autoresearch/<tag>            # then point a fresh Claude session at autoresearch/program.md
python autoresearch/final_eval.py --tag <tag> --n-experiments <N> --wall-clock-hours <H>   # human-only, after the loop
```

## Rules that keep the comparison fair

- The agent never sees the test set, `first_date`, or the conventional pipeline's code and results. This is enforced by instruction in `program.md`, not technically, so run the agent with permissions that keep it out of those paths.
- The test set is used once, by `final_eval.py`.
- Metrics are written as JSON to `results/`, in the same shape as `results/conventional_xgboost.json`.

## Known limitation

A training customer's churn label looks 90 days past their first purchase, so the last customers in each training fold are labelled using weeks that overlap the start of the validation window. The conventional pipeline has the same overlap, so the comparison stays like-for-like, but absolute AUCs may be somewhat optimistic.
