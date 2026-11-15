"""The ONE file the autoresearch agent edits (approach 3: raw line items).

``make_features`` turns line items into one row per customer; ``make_model`` is the
XGBoost classifier. Starting point: five mechanical per-customer roll-ups and a
plain default XGBoost. Everything else (more features, encodings, feature
selection, hyperparameters, early stopping) is for the agent to find.

Keep ALL feature engineering inside ``make_features``. Anything learned from data
must be learned from ``lines_fit`` / ``y_fit`` only and applied to ``lines_apply``.
The model must be XGBoost (``xgboost.XGBClassifier``).
"""
import pandas as pd
import xgboost as xgb

from prepare import RANDOM_SEED, run_cv


def make_features(lines_fit, y_fit, lines_apply):
    def rollup(lines):
        lines = lines.assign(line_total=lines['quantity'] * lines['price'])
        return lines.groupby('customer_id').agg(
            n_lines=('stock_code', 'size'),
            n_distinct_products=('stock_code', 'nunique'),
            total_quantity=('quantity', 'sum'),
            total_spend=('line_total', 'sum'),
            avg_unit_price=('price', 'mean'),
        )

    return rollup(lines_fit), rollup(lines_apply)


def make_model():
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='auc',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


if __name__ == '__main__':
    result = run_cv(make_features, make_model)
    print(f"Fold AUCs: {[round(a, 4) for a in result['fold_aucs']]}")
    print(f"CV time: {result['seconds']:.1f}s")
    print(f"CV AUC: {result['cv_auc']:.4f}")
