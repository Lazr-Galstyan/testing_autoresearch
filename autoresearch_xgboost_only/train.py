"""The ONE file the autoresearch agent edits.

Starting point: a plain XGBoost with default-ish settings on the shared features
(``data/processed/train.csv``), no feature engineering. Everything below is fair
game: feature engineering inside ``make_features``, the XGBoost hyperparameters
inside ``make_model``, early stopping, etc. The predictor must be XGBoost only:
a single ``XGBClassifier``, or several XGBoost models combined. No other model
families.

Keep ALL feature engineering inside ``make_features`` (no helpers elsewhere that
``final_eval.py`` would not pick up). Anything learned from data must be learned
from ``X_fit`` only and applied to ``X_apply``.
"""
import xgboost as xgb

from prepare import RANDOM_SEED, run_cv


def make_features(X_fit, X_apply):
    return X_fit, X_apply


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
