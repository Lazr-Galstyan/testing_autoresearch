# Data

## Structure

```
data/
├── raw/        # Original, unmodified source file — never edit it
└── processed/  # Tables written by the scripts in src/ (git-ignored)
```

## Dataset

**[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
from the UCI Machine Learning Repository (Daqing Chen et al., 2012).

Transactions of a UK-based online retailer, 2009-12-01 to 2011-12-09, about 1.07M rows, one per invoice line item. There is no churn label in the source.

### Columns

| Column | Description |
|---|---|
| `Invoice` | Invoice number. Prefixed with `C` for cancellations. |
| `StockCode` | Product code. |
| `Description` | Product name. |
| `Quantity` | Units on the line. Negative for cancellations/returns. |
| `InvoiceDate` | Invoice date and time. |
| `Price` | Unit price in pounds sterling. |
| `Customer ID` | Customer number. Missing for non-attributed transactions. |
| `Country` | Customer's country. |

### How to Obtain

1. Go to the [UCI dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
2. Download the data.
3. Save it as `data/raw/online_retail_II.csv`, the path `src/prepare_churn_data.py` reads.

## Churn label

Built by `src/prepare_churn_data.py`. Each customer is represented by their first transaction. `churn = 1` if they made no second purchase within 90 days of the first (or never made one), otherwise `0`. Customers whose first purchase is within the last 90 days of the data are dropped, because their label cannot yet be determined.

## Processed files (in `data/processed/`)

| File | Written by | Content |
|---|---|---|
| `first_transaction_churn.csv` | `prepare_churn_data.py` | Line items of each customer's first invoice, plus `churn` |
| `first_transaction_churn_clean.csv` | `correct_data_issues.py` | The same after data-quality corrections |
| `item_history.csv`, `description_embeddings.npz` | `build_item_history.py`, `description_embeddings.py` | Product lookup tables |
| `customer_features.csv` | `build_features.py` | Customer-grain features |
| `modelling_dataset.csv` | `build_modelling_dataset.py` | One row per customer, with unreduced embeddings |
| `train.csv`, `test.csv`, `pca.joblib` | `split_data.py` | Time-ordered 80/20 split, embeddings reduced with PCA fitted on train; used by approaches 1 and 2 |

## Note

Raw and processed files are not tracked by git (see `.gitignore`). Follow the steps above to reproduce them locally.
