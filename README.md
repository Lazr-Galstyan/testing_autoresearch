# Testing Autoresearch on E-Commerce Churn

A benchmarking project that compares **Andrej Karpathy's autoresearch framework** against a **traditional XGBoost pipeline** on an e-commerce customer churn dataset.

## Motivation

Traditional ML workflows require significant manual effort: hand-crafted features, iterative hyperparameter searches, and researcher intuition guiding each step. Karpathy's autoresearch proposes automating much of this loop — hypothesis generation, feature discovery, and model refinement — through an agentic research process. This project measures whether that automation translates into better predictive performance with less human effort.

## Approaches Compared

Three pipelines are benchmarked. All use XGBoost as the base model and the same time-ordered train/test split, so the held-out test metrics are comparable.

| | 1. Conventional | 2. Autoresearch, shared features | 3. Autoresearch, raw data |
|---|---|---|---|
| Status | Done | Scaffold built, loop not yet run | Planned, not implemented |
| Directory | `conventional_approach/`, `src/xgboost_tuning.py` | `autoresearch/` | `autoresearch/` (TBD) |
| Input | `data/processed/train.csv` and `test.csv` (hand-built features from `src/`) | The same `train.csv` and `test.csv` as approach 1 | `data/raw/online_retail_II.csv`, the original line-item transactions |
| Feature engineering | Manual (the `src/` pipeline) | Agent builds on top of the shared features | Agent builds all features from the raw line items |
| Churn label and split | Hand-built | Supplied (same as 1) | Supplied: the churn label and the train/test customer split are given; the agent builds the features |
| Hyperparameter tuning | Optuna TPE, 100 trials, time-ordered CV | Agent-driven, same time-ordered CV | Agent-driven |
| Researcher effort | High | Low | Low |

**Approach 1, Conventional.** Hand-engineered customer features (`src/`), then XGBoost tuned with Optuna against mean AUC-ROC over four expanding-window, time-ordered folds, fitted on the full training set and evaluated once on the test set. Output: `results/conventional_xgboost.json`.

**Approach 2, Autoresearch on the shared features.** An LLM agent edits `autoresearch/train.py` in a keep-or-discard loop, scored by the same four folds. It tests how much an agent adds on top of the conventional feature set. See `autoresearch/README.md`. Output: `results/autoresearch_shared_features.json`.

**Approach 3, Autoresearch on raw data (planned).** The agent starts from the untransformed transaction file instead of the engineered features. It is given the churn label and the train/test customer split, and has to discover the features itself. It tests whether an agent can replace the hand-built feature pipeline, not just improve on it. Design details are not yet decided.

## Dataset

**[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
from the UCI Machine Learning Repository (Daqing Chen, Chen et al. 2012).

Transactional records from a UK-based, registered non-store online retailer covering 2009-12-01 to 2011-12-09. Roughly 1.07M rows, one row per line item on an invoice. The dataset **does not include a churn label** — churn is defined downstream from the transactional history: **churn = 1** if a customer makes no second purchase within 90 days of their first purchase (or no second purchase at all), else 0. Customers whose first purchase is within the last 90 days of the data are dropped because their label cannot yet be known (`src/prepare_churn_data.py`).

| Column | Description |
|---|---|
| `Invoice` | Invoice number (6-digit). Prefixed with `C` for cancellations. |
| `StockCode` | Product (item) code. |
| `Description` | Product name. |
| `Quantity` | Units of the product per transaction line. Negative for cancellations/returns. |
| `InvoiceDate` | Invoice date and time. |
| `Price` | Unit price in pounds sterling (£). |
| `Customer ID` | 5-digit customer number. May be missing for guest / non-attributed transactions. |
| `Country` | Country where the customer resides. |

## External References

- Autoresearch by Andrej Karpathy: [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- XGBoost adaptation of autoresearch (Pafka and Ariño de la Rubia): [github.com/szilard/xgboost-autoresearch](https://github.com/szilard/xgboost-autoresearch)
- Dataset: [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

## Project Structure

```
testing_autoresearch/
├── data/                   # raw/ (git-ignored) and processed/ (git-ignored)
├── src/                    # data cleaning, feature engineering, split, and xgboost_tuning.py
├── conventional_approach/  # Approach 1: notebook calling src/xgboost_tuning.py
├── autoresearch/           # Approach 2 (and later 3): agent harness, train.py, program.md
├── notebooks/              # EDA, data-quality checks, feature-engineering walkthrough
├── results/                # Metrics as JSON
└── README.md
```

## Reproducing Results

1. Clone the repository and install dependencies: `pip install -r requirements.txt`
2. Download the dataset into `data/raw/` (see `data/README.md`).
3. Build the data (each script reads the previous one's output):
   ```bash
   python src/prepare_churn_data.py      # first transactions + churn label
   python src/correct_data_issues.py     # data-quality corrections
   python src/build_modelling_dataset.py # one row per customer
   python src/split_data.py              # time-ordered train/test + PCA
   ```
4. Approach 1: run `conventional_approach/xgboost_model.ipynb` (logic in `src/xgboost_tuning.py`).
5. Approach 2: see `autoresearch/README.md`.
6. Approach 3: not yet implemented.
7. Compare the JSON files in `results/`.

## Goals

- Quantify autoresearch's improvement over a traditional approach (AUC-ROC, F1, precision/recall)
- Measure researcher time-to-result for each approach
- Identify which parts of the ML pipeline autoresearch improves most
