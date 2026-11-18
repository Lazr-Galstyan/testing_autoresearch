# Autoresearch on the shared features, XGBoost only

Same task, data and harness as `autoresearch/` (the processed `train.csv`, the same four
time-ordered CV folds), but the agent is restricted to **XGBoost**. It can engineer features
and tune XGBoost (and may combine several XGBoost models), but it may not use any other model
family. This isolates what an agent adds through features and XGBoost settings alone, and
separates it from the gain `autoresearch/` got from blending in a logistic regression and a
random forest.

## How it differs from `autoresearch/`

| | `autoresearch/` | this folder |
|---|---|---|
| Predictor | Any model | XGBoost only (single model or XGBoost blends) |
| `program.md` | Written for this project | Built on the instructions of [szilard/xgboost-autoresearch](https://github.com/szilard/xgboost-autoresearch) (research-before-guessing, hypothesis and classification per experiment, plateau rule, synthesis every 10 experiments), adapted to this project's harness |
| Log | experiment, model, modification, CV_AUC, cv_seconds, status | adds a `variables` column with the complete feature list per experiment |
| Starting point | Default XGBoost, no feature engineering | The same |

The harness (`prepare.py`) is identical to the one in `autoresearch/`, so the scores are comparable. `final_eval.py` differs only in writing `results/autoresearch_xgboost_only.json`.

## Files

| File | Edited by | Purpose |
|---|---|---|
| `prepare.py` | nobody (fixed) | Loads `train.csv` only, builds the time-ordered folds, runs `run_cv`. Never loads the test set. |
| `train.py` | the agent | `make_features(X_fit, X_apply)` and `make_model()`; starts as default XGBoost. |
| `program.md` | the human | The agent's instructions: 100 experiments, no commits, XGBoost only. |
| `final_eval.py` | nobody (human-only) | Fits on all of train, evaluates once on test, writes `results/autoresearch_xgboost_only.json`. |
| `results.tsv`, `research-log.md`, `best_train.py` | the agent | Experiment log, reasoning log, backup of the best `train.py`. |
| `run.log` | the agent | Latest run output. |

## Running it

```bash
python3 autoresearch_xgboost_only/train.py     # sanity check: baseline CV AUC 0.5712
# then start a fresh Claude Code session in an isolated copy and point it at autoresearch_xgboost_only/program.md
python autoresearch_xgboost_only/final_eval.py --tag <tag> --n-experiments 100 --wall-clock-hours <H>   # human-only, after the loop
```

The agent must not see the test set or the other approaches' results. That is an instruction in `program.md`, not a technical block, so run it in an isolated copy that contains only `data/processed/train.csv`.

## Known limitation

A training customer's label looks 90 days past their first purchase, so the last customers in each training fold are labelled using weeks that overlap the start of the validation window. All approaches have the same overlap.
