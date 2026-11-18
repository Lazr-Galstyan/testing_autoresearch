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
import numpy as np
import pandas as pd
import xgboost as xgb

from prepare import RANDOM_SEED, run_cv


DRIFTING_PREFIXES = ('prior_transactions_', 'prior_units_', 'qty_share_of_prior_units_', 'year')
LINEAR_ONLY_PREFIXES = ('country_', 'weekday_', 'hour', 'time_of_day_', 'rate_')   # noise for trees, additive for linear
N_MEMBERS = 40
LINEAR_WEIGHT = 0.7


def make_features(X_fit, X_apply):
    drop = [c for c in X_fit.columns if c.startswith(DRIFTING_PREFIXES)
            or (c.startswith(('prior_', 'qty_vs_', 'price_vs_')) and c.endswith('_sum'))]
    cumulative = [c for c in X_fit.columns if c.startswith(('prior_transactions_', 'prior_units_'))]

    def day(X):
        return pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day_of_month']))

    start = day(X_fit).min()

    def build(X):   # cumulative popularity counts / days of history -> per-day rates (linear member only)
        rates = X[cumulative].div((day(X) - start).dt.days + 30, axis=0).add_prefix('rate_')
        return X.drop(columns=drop).join(rates)

    return build(X_fit), build(X_apply)


def base_model(seed):
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        min_child_weight=20,
        max_bin=64,
        eval_metric='auc',
        random_state=seed,
        n_jobs=-1,
    )


class RotatedPCAEnsemble:
    """Average of XGBoost members, each trained on its own random orthogonal rotation of the pca_* block."""

    def fit(self, X, y):
        X_all = X
        self.linear_only = [c for c in X.columns if c.startswith(LINEAR_ONLY_PREFIXES)]
        X = X.drop(columns=self.linear_only)
        self.pca_cols = [c for c in X.columns if c.startswith('pca_')]
        self.hist_cols = [c for c in X.columns if c.startswith(('prior_', 'qty_vs_', 'price_vs_'))]
        h = self._slog(X[self.hist_cols].to_numpy(dtype=float))
        self.mu, self.sd = np.nanmean(h, axis=0), np.nanstd(h, axis=0) + 1e-9
        p = X[self.pca_cols].to_numpy()
        self.p_mu, self.p_sd = p.mean(axis=0), p.std(axis=0) + 1e-9
        rng = np.random.default_rng(RANDOM_SEED)
        self.members = []
        n = len(self.pca_cols) + len(self.hist_cols)
        for k in range(N_MEMBERS):
            q, _ = np.linalg.qr(rng.standard_normal((n, n)))
            self.members.append((q, base_model(RANDOM_SEED + k).fit(self._rotate(X, q), y)))
        X = X_all
        # linear XGBoost member on signed-log z-scored inputs (stats from fit rows)
        a = self._slog(X.to_numpy(dtype=float))
        self.l_mu, self.l_sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0) + 1e-9
        self.linear = xgb.XGBClassifier(booster='gblinear', n_estimators=200, learning_rate=0.1, reg_lambda=0.1,
                                        reg_alpha=0.01, random_state=RANDOM_SEED, n_jobs=-1)
        self.linear.fit(self._lin(X), y)
        return self

    def _lin(self, X):
        return np.nan_to_num((self._slog(X.to_numpy(dtype=float)) - self.l_mu) / self.l_sd)

    @staticmethod
    def _slog(a):
        return np.sign(a) * np.log1p(np.abs(a))

    def _rotate(self, X, q):
        X = X.copy()
        p = (X[self.pca_cols].to_numpy() - self.p_mu) / self.p_sd
        z = np.nan_to_num((self._slog(X[self.hist_cols].to_numpy(dtype=float)) - self.mu) / self.sd)
        X[self.pca_cols + self.hist_cols] = np.hstack([p, z]) @ q
        return X

    def predict_proba(self, X):
        Xt = X.drop(columns=self.linear_only)
        trees = np.mean([m.predict_proba(self._rotate(Xt, q)) for q, m in self.members], axis=0)
        return (1 - LINEAR_WEIGHT) * trees + LINEAR_WEIGHT * self.linear.predict_proba(self._lin(X))


def make_model():
    return RotatedPCAEnsemble()


if __name__ == '__main__':
    result = run_cv(make_features, make_model)
    print(f"Fold AUCs: {[round(a, 4) for a in result['fold_aucs']]}")
    print(f"CV time: {result['seconds']:.1f}s")
    print(f"CV AUC: {result['cv_auc']:.4f}")
