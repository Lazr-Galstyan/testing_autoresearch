"""The ONE file the autoresearch agent edits (approach 3: raw line items).

``make_features`` turns line items into one row per customer; ``make_model`` is the
XGBoost classifier. Starting point: five mechanical per-customer roll-ups and a
plain default XGBoost. Everything else (more features, encodings, feature
selection, hyperparameters, early stopping) is for the agent to find.

Keep ALL feature engineering inside ``make_features``. Anything learned from data
must be learned from ``lines_fit`` / ``y_fit`` only and applied to ``lines_apply``.
The model must be XGBoost (``xgboost.XGBClassifier``).
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from prepare import RANDOM_SEED, run_cv

TE_SMOOTHING = 1000
TE_FOLDS = 5


def product_te(lines_fit, y_fit, lines_apply):
    """Basket mean of each product's de-trended churn effect, learned from fit customers only.

    Churn is first de-trended by the fit mean of the customer's first-invoice week, so a product's
    encoding is its effect relative to its season. Fit customers get out-of-fold values.
    """
    cohort = lines_fit.groupby('customer_id')['invoice_date'].min().dt.to_period('W')
    resid = y_fit - y_fit.groupby(cohort.loc[y_fit.index]).transform('mean')

    def encode(source, target):
        source = source[['customer_id', 'stock_code']].drop_duplicates()
        stats = source.assign(r=source['customer_id'].map(resid)).groupby('stock_code')['r'].agg(['sum', 'count'])
        te = stats['sum'] / (stats['count'] + TE_SMOOTHING)
        return target.assign(te=target['stock_code'].map(te)).groupby('customer_id')['te'].mean()

    fold = pd.Series(np.random.default_rng(RANDOM_SEED).integers(0, TE_FOLDS, len(y_fit)), index=y_fit.index)
    line_fold = lines_fit['customer_id'].map(fold)
    te_fit = pd.concat([encode(lines_fit[line_fold != k], lines_fit[line_fold == k]) for k in range(TE_FOLDS)])
    return te_fit, encode(lines_fit, lines_apply)


def make_features(lines_fit, y_fit, lines_apply):
    def rollup(lines):
        lines = lines.assign(line_total=lines['quantity'] * lines['price'])
        return lines.groupby('customer_id').agg(
            n_distinct_products=('stock_code', 'nunique'),
            total_spend=('line_total', 'sum'),
            avg_unit_price=('price', 'mean'),
            median_quantity=('quantity', 'median'),
            max_quantity=('quantity', 'max'),
            std_unit_price=('price', 'std'),
        )

    te_fit, te_apply = product_te(lines_fit, y_fit, lines_apply)
    return rollup(lines_fit).assign(prod_te_mean=te_fit), rollup(lines_apply).assign(prod_te_mean=te_apply)


def make_model():
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.5,
        colsample_bytree=0.6,
        min_child_weight=2,
        reg_lambda=20,
        eval_metric='auc',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


if __name__ == '__main__':
    result = run_cv(make_features, make_model)
    print(f"Fold AUCs: {[round(a, 4) for a in result['fold_aucs']]}")
    print(f"CV time: {result['seconds']:.1f}s")
    print(f"CV AUC: {result['cv_auc']:.4f}")
