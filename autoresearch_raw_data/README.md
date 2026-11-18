# Autoresearch on raw line items (approach 3)

Same loop as `autoresearch/`, but the agent starts from the **raw transaction line items** instead of the hand-engineered feature table. It has to build every customer feature itself. It tests whether an agent can replace the hand-built feature pipeline, not just improve on it.

## The data

`data/processed/raw_train.csv` and `raw_test.csv` (git-ignored, built by `build_data.py`): one row per item on each customer's **first invoice**, after the data-quality corrections of `src/correct_data_issues.py`, with the columns

`invoice, stock_code, description, quantity, invoice_date, price, customer_id, country, churn`

and nothing derived. `churn` is the per-customer label (no second purchase within 90 days of the first), repeated on each of the customer's lines. Only first invoices are included, because later purchases would reveal the label.

The customers and the train/test split are exactly those of approaches 1 and 2 (4,035 train and 1,009 test customers, cutoff `2010-11-30 16:29`), and the CV folds are the same four time-ordered expanding windows (training on the earliest 60 / 70 / 80 / 90 % of customers: 2421 / 2824 / 3227 / 3632 customers). `build_data.py` checks the customer sets, sizes and first-purchase times against the existing split.

## What the agent does

It edits `train.py`: `make_features(lines_fit, y_fit, lines_apply)` turns line items into one row per customer, and `make_model()` returns an XGBoost classifier. The baseline has five mechanical roll-ups (line count, distinct products, total quantity, total spend, mean unit price) and a default XGBoost. The model must stay XGBoost, so only the features and XGBoost settings change. It runs exactly 100 experiments, baseline included, with no git commits; `best_train.py` is the backup used to undo a bad change.

## Files

| File | Edited by | Purpose |
|---|---|---|
| `build_data.py` | nobody (human-only) | Builds `raw_train.csv` / `raw_test.csv` from `data/processed/` and checks them against the existing split. |
| `prepare.py` | nobody (fixed) | Loads `raw_train.csv` only, builds the customer table and folds, runs `run_cv`. Never loads the test set. |
| `train.py` | the agent | Feature engineering and the XGBoost model. |
| `program.md` | the human | The agent's instructions. |
| `final_eval.py` | nobody (human-only) | Fits on all of train, evaluates once on test, writes `results/autoresearch_raw_data.json`. |
| `results.tsv`, `research-log.md`, `best_train.py` | the agent | Experiment log, reasoning log, backup of the best `train.py`. Tracked in git. |
| `run.log` | the agent | Latest run output. Git-ignored. |

## Running it

```bash
python autoresearch_raw_data/build_data.py     # once, needs data/processed/ from the earlier steps
python3 autoresearch_raw_data/train.py         # sanity check: baseline CV AUC 0.5947
# then start a fresh Claude Code session in an isolated copy and point it at autoresearch_raw_data/program.md
python autoresearch_raw_data/final_eval.py --tag <tag> --n-experiments 100 --wall-clock-hours <H>   # human-only, after the loop
```

The agent must not see `raw_test.csv`, the engineered data, or the results of the other approaches. That is an instruction in `program.md`, not a technical block, so run it in an isolated copy that contains only `raw_train.csv`.

## Known limitation

A training customer's label looks 90 days past their first purchase, so the last customers in each training fold are labelled using weeks that overlap the start of the validation window. All three approaches have the same overlap.
