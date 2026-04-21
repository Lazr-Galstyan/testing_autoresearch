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

**[E-commerce Customer Data For Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)**
by Shriyash Jagtap on Kaggle.

An e-commerce transaction dataset designed for customer churn prediction and behavior analysis. The target variable is binary churn (0 = retained, 1 = churned).

| Column | Description |
|---|---|
| `Customer ID` | Unique customer identifier |
| `Customer Name` | Synthetic customer name |
| `Customer Age` | Age of the customer |
| `Gender` | Customer gender |
| `Purchase Date` | Timestamp of the transaction |
| `Product Category` | Category of the purchased item |
| `Product Price` | Price of the individual product |
| `Quantity` | Units purchased per transaction |
| `Total Purchase Amount` | Total transaction value |
| `Payment Method` | Payment method used (e.g. credit card, PayPal) |
| `Returns` | Whether the item was returned (0 = no, 1 = yes) |
| `Churn` | **Target** — whether the customer churned (0 = retained, 1 = churned) |

Churn is inferred from purchase frequency, recency, and platform interaction patterns.

## External References

- Autoresearch project by Andrej Karpathy: *(link TBD)*
- Dataset: [Kaggle — E-commerce Customer Data For Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)

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
