# Data

## Structure

```
data/
├── raw/        # Original, unmodified source files — never edit these
└── processed/  # Cleaned/encoded outputs written by the preprocessing notebook
```

## Dataset

**[E-commerce Customer Data For Behavior Analysis](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)**
by Shriyash Jagtap — Kaggle

### Columns

| Column | Type | Description |
|---|---|---|
| `Customer ID` | int | Unique customer identifier |
| `Customer Name` | str | Synthetic customer name |
| `Customer Age` | int | Age of the customer |
| `Gender` | str | Customer gender |
| `Purchase Date` | datetime | Transaction timestamp |
| `Product Category` | str | Category of the purchased item |
| `Product Price` | float | Price of the individual product |
| `Quantity` | int | Units purchased per transaction |
| `Total Purchase Amount` | float | Total transaction value |
| `Payment Method` | str | Payment method used (e.g. credit card, PayPal) |
| `Returns` | int | Item returned: 0 = no, 1 = yes |
| `Churn` | int | **Target** — 0 = retained, 1 = churned |

### How to Obtain

1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis?resource=download)
2. Download the Excel file
3. Place it in `data/raw/`

## Note

Raw data files are not tracked by git (see `.gitignore`). Follow the retrieval steps above to reproduce the dataset locally.
