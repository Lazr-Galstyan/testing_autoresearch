# Testing Autoresearch on E-Commerce Churn

A benchmarking project that compares **Andrej Karpathy's autoresearch framework** against a **traditional XGBoost pipeline** on an e-commerce customer churn dataset.

## Motivation

Traditional ML workflows require significant manual effort: hand-crafted features, iterative hyperparameter searches, and researcher intuition guiding each step. Karpathy's autoresearch proposes automating much of this loop — hypothesis generation, feature discovery, and model refinement — through an agentic research process. This project measures whether that automation translates into better predictive performance with less human effort.

## Approaches Compared

| | Traditional | Autoresearch |
|---|---|---|
| Feature engineering | Manual | Automated |
| Hyperparameter tuning | Grid / random search | Automated |
| Model | XGBoost | XGBoost (same base) |
| Researcher effort | High | Low |

## Dataset

**[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
from the UCI Machine Learning Repository (Daqing Chen, Chen et al. 2012).

Transactional records from a UK-based, registered non-store online retailer covering 2009-12-01 to 2011-12-09. Roughly 1.07M rows, one row per line item on an invoice. The dataset **does not include a churn label** — churn is defined downstream from the transactional history (rule TBD; see `notebooks/01_eda_raw.ipynb`).

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

- Autoresearch project by Andrej Karpathy: *(link TBD)*
- Dataset: [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

## Project Structure

```
testing_autoresearch/
├── data/              # Raw and processed datasets
├── traditional/       # Baseline XGBoost pipeline
├── autoresearch/      # Karpathy autoresearch pipeline
├── notebooks/         # EDA and comparison analyses
├── results/           # Metrics, plots, model artefacts
└── README.md
```

## Reproducing Results

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset and place it in `data/` *(instructions TBD)*
4. Run the traditional pipeline: `python traditional/train.py`
5. Run the autoresearch pipeline: `python autoresearch/run.py`
6. Compare results in `results/`

## Goals

- Quantify autoresearch's improvement over a traditional approach (AUC-ROC, F1, precision/recall)
- Measure researcher time-to-result for each approach
- Identify which parts of the ML pipeline autoresearch improves most
