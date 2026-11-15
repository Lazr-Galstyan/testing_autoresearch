"""The ONE file the autoresearch agent edits.

Starting point: a plain XGBoost with default-ish settings on the shared features
(``data/processed/train.csv``), no feature engineering. Everything below is fair
game: feature engineering inside ``make_features``, the model and its
hyperparameters inside ``make_model``, early stopping, etc.

Keep ALL feature engineering inside ``make_features`` (no helpers elsewhere that
``final_eval.py`` would not pick up). Anything learned from data must be learned
from ``X_fit`` only and applied to ``X_apply``.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from prepare import RANDOM_SEED, run_cv

CORE_FEATURES = [
    'total_spend', 'total_quantity', 'n_distinct_products', 'n_lines',
    'prior_median_daily_transactions_mean', 'prior_median_daily_transactions_max',
    'prior_median_daily_units_mean', 'prior_median_daily_units_max',
    'avg_unit_price', 'day_of_month', 'hour',
]


def make_features(X_fit, X_apply):
    return X_fit, X_apply


def make_model():
    linear = make_pipeline(
        SimpleImputer(strategy='median'),
        QuantileTransformer(output_distribution='normal', n_quantiles=200,
                            random_state=RANDOM_SEED),
        LogisticRegression(C=0.3, max_iter=2000),
    )
    core = make_pipeline(
        ColumnTransformer([('core', 'passthrough', CORE_FEATURES)]),
        SimpleImputer(strategy='median'),
        QuantileTransformer(output_distribution='normal', n_quantiles=200,
                            random_state=RANDOM_SEED),
        LogisticRegression(C=10.0, max_iter=2000),
    )
    return VotingClassifier([('wide', linear), ('core', core)], voting='soft',
                            weights=[1.0, 4.0])


if __name__ == '__main__':
    result = run_cv(make_features, make_model)
    print(f"Fold AUCs: {[round(a, 4) for a in result['fold_aucs']]}")
    print(f"CV time: {result['seconds']:.1f}s")
    print(f"CV AUC: {result['cv_auc']:.4f}")
