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

## Comparison Axes
| Dimension | Metric |
|---|---|
| Predictive performance | AUC-ROC, F1, precision/recall on held-out test set |
| Efficiency | Time-to-result (wall clock) |
| Effort | Lines of code, manual decisions required |

## Coding Conventions
- Python 3.10+
- Dependencies: `requirements.txt` or `pyproject.toml`
- Always set `RANDOM_SEED = 42` explicitly — reproducibility is non-negotiable
- Store all metrics as JSON in `results/` for programmatic comparison
- Use `pathlib.Path` relative to project root — never hardcode paths
- Notebooks for exploration only; production code in `.py` modules
- Do not commit raw data files >5MB; document retrieval steps in `data/README.md`

## How Claude Should Help
- When writing new code, follow existing patterns in `traditional/` or `autoresearch/`
- If asked to add a feature, also add the corresponding metric logging to `results/`
- Prefer clarity over cleverness — this is benchmarking code, not production
- When both pipelines need the same change (e.g. a new eval metric), apply it to both
- Always evaluate on the held-out test set; never report validation metrics as final

## What to Avoid
- Do not mix autoresearch and traditional pipeline code
- Do not skip the held-out test set when reporting final numbers
- Do not add dependencies without updating `requirements.txt`
