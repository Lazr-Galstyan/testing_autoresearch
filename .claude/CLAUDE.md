# Project Context

## What This Project Does

This project benchmarks Andrej Karpathy's **autoresearch** framework against a **traditional XGBoost pipeline** on an e-commerce customer churn dataset. The goal is to empirically demonstrate where and how autoresearch outperforms the conventional approach in terms of feature engineering, model selection, and overall predictive performance.

## Project Structure (expected)

```
testing_autoresearch/
├── data/                  # Raw and processed churn datasets
├── traditional/           # Baseline XGBoost pipeline (manual feature engineering, grid search)
├── autoresearch/          # Karpathy autoresearch pipeline
├── notebooks/             # EDA, comparison analyses, result visualisations
├── results/               # Saved metrics, plots, model artefacts
└── README.md
```

## Key Concepts

- **Traditional approach**: manual feature engineering → XGBoost → hyperparameter tuning via grid/random search → evaluation
- **Autoresearch approach**: automated hypothesis generation, feature discovery, and iterative model refinement as described by Karpathy
- **Comparison axis**: AUC-ROC, F1, precision/recall on held-out test set; time-to-result; researcher effort

## Data

E-commerce churn dataset. Links TBD — will be added to `data/README.md` once provided.

## External References

- Autoresearch project: link TBD
- Dataset source: link TBD

## Coding Conventions

- Python 3.10+
- Dependencies managed with `requirements.txt` or `pyproject.toml`
- All experiments should be reproducible: set random seeds explicitly (`RANDOM_SEED = 42`)
- Store metrics as JSON in `results/` so they can be compared programmatically
- Notebooks are for exploration only; production-ready code lives in `.py` modules

## What to Avoid

- Do not hardcode file paths — use `pathlib.Path` relative to the project root
- Do not commit raw data files larger than a few MB; use `.gitignore` and document how to obtain them
- Do not skip evaluation on the held-out test set when reporting final numbers
