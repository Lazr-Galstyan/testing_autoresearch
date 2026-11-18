"""The ONE file the autoresearch agent edits.

Starting point: a plain XGBoost with default-ish settings on the shared features
(``data/processed/train.csv``), no feature engineering. Everything below is fair
game: feature engineering inside ``make_features``, the model and its
hyperparameters inside ``make_model``, early stopping, etc.

Keep ALL feature engineering inside ``make_features`` (no helpers elsewhere that
``final_eval.py`` would not pick up). Anything learned from data must be learned
from ``X_fit`` only and applied to ``X_apply``.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from prepare import RANDOM_SEED, run_cv


DRIFTING_PREFIXES = ('prior_transactions_', 'prior_units_', 'qty_share_of_prior_units_', 'year', 'month', 'time_of_day_')


def make_features(X_fit, X_apply):
    drop = [c for c in X_fit.columns if c.startswith(DRIFTING_PREFIXES)]
    months = sorted(X_fit['month'].unique())   # month levels seen in the fit rows only
    cumulative = [c for c in X_fit.columns if c.startswith(('prior_transactions_', 'prior_units_'))]

    def day(X):
        return pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day_of_month']))

    start = day(X_fit).min()

    def build(X):   # cumulative popularity counts / days of history -> per-day rates (no calendar drift)
        rates = X[cumulative].div((day(X) - start).dt.days + 30, axis=0).add_prefix('rate_')
        return X.drop(columns=drop).join(rates).join(pd.DataFrame(
            {f'month_{m}': (X['month'] == m).astype(float) for m in months}, index=X.index)).assign(
            days_to_christmas=(pd.to_datetime(dict(year=X['year'], month=12, day=25)) - day(X)).dt.days % 365,
            doy_sin=np.sin(2 * np.pi * day(X).dt.dayofyear / 365.25),
            doy_cos=np.cos(2 * np.pi * day(X).dt.dayofyear / 365.25),
            elapsed_days=(day(X) - start).dt.days)   # sample weights + time trend

    return build(X_fit), build(X_apply)


def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


HALF_LIFE_DAYS = 210


class RecencyWeighted:
    """Fit the linear pipeline with sample weights halving every HALF_LIFE_DAYS before the latest fit row."""

    def fit(self, X, y):
        e = X['elapsed_days'].to_numpy()
        w = 0.5 ** ((e.max() - e) / HALF_LIFE_DAYS)
        self.model = linear_model().fit(self._years(X), y,
                                        baggingclassifier__sample_weight=w / w.mean())
        return self

    @staticmethod
    def _years(X):   # elapsed time in years: signed_log1p is then close to linear over the data range
        return X.assign(elapsed_days=X['elapsed_days'] / 3650)

    def predict_proba(self, X):   # average the members' logits (geometric pool) instead of their probabilities
        p = 1 / (1 + np.exp(-self.model.decision_function(self._years(X))))
        return np.c_[1 - p, p]


def make_model():
    return RecencyWeighted()


def linear_model():
    return make_pipeline(
        FunctionTransformer(signed_log),
        SimpleImputer(strategy='constant', fill_value=0, add_indicator=True),
        StandardScaler(),
        BaggingClassifier(LogisticRegression(C=1.0, max_iter=2000), n_estimators=50, max_features=0.4, bootstrap=False,
                          random_state=RANDOM_SEED, n_jobs=-1),
    )


if __name__ == '__main__':
    result = run_cv(make_features, make_model)
    print(f"Fold AUCs: {[round(a, 4) for a in result['fold_aucs']]}")
    print(f"CV time: {result['seconds']:.1f}s")
    print(f"CV AUC: {result['cv_auc']:.4f}")
