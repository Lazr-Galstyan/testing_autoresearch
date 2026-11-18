"""Fixed evaluation harness for the autoresearch loop (approach 1: shared features).

The agent MUST NOT edit this file. It owns everything that has to stay constant
between experiments so that scores are comparable:

* the data: ``data/processed/train.csv`` only. The test set is never loaded here;
  only ``final_eval.py`` (human-only) touches it.
* the cross-validation folds: identical to ``src/xgboost_tuning.py`` (time-ordered,
  expanding window, train fractions 0.6/0.7/0.8/0.9), so the conventional and
  autoresearch scores are computed the same way.
* the objective: mean validation AUC-ROC over those folds.

The agent's ``train.py`` supplies two functions, ``make_features`` and
``make_model``; ``run_cv`` below calls them once per fold.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
TRAIN_FRACTIONS = (0.6, 0.7, 0.8, 0.9)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'

LABEL_COLUMN = 'churn'
DATE_COLUMN = 'first_date'


def load_train() -> pd.DataFrame:
    """Training customers, in file order, with ``churn`` and ``first_date`` kept."""
    return pd.read_csv(TRAIN_FILE, parse_dates=[DATE_COLUMN])


def time_ordered_folds(dates: pd.Series, train_fractions=TRAIN_FRACTIONS):
    """Expanding-window folds; same logic as ``src/xgboost_tuning.time_ordered_folds``.

    Returns a list of ``(train_positions, validation_positions)`` arrays. Training
    is every row strictly before the cutoff date, validation every row at or after.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    ordered = dates.sort_values(kind='stable').reset_index(drop=True)
    folds = []
    for fraction in train_fractions:
        cutoff = ordered.iloc[int(round(len(ordered) * fraction))]
        folds.append((np.flatnonzero(dates < cutoff), np.flatnonzero(dates >= cutoff)))
    return folds


def split_xy(df: pd.DataFrame):
    """Separate the label and the date from the raw feature columns.

    ``first_date`` is only for ordering rows; it is NOT passed to the agent's code
    as a column, so it cannot become a feature (the date-derived columns already
    in the data, such as ``year`` and ``month``, are the agent's to use).
    """
    return df.drop(columns=[LABEL_COLUMN, DATE_COLUMN]), df[LABEL_COLUMN]


def run_cv(make_features, make_model) -> dict:
    """Score the agent's pipeline with the fixed time-ordered CV.

    Input:
        ``make_features(X_fit, X_apply) -> (X_fit', X_apply')``: any learned
        state (category levels, scalers, target statistics, ...) must come from
        ``X_fit`` only. It is called per fold with that fold's training rows as
        ``X_fit`` and its validation rows as ``X_apply``, so nothing from the
        validation window can leak in.
        ``make_model() -> estimator``: a fresh unfitted model with
        ``fit(X, y)`` and ``predict_proba(X)``.
    Output:
        ``{'cv_auc': mean, 'fold_aucs': [...], 'seconds': wall clock}``.
    """
    df = load_train()
    X, y = split_xy(df)
    folds = time_ordered_folds(df[DATE_COLUMN])

    start = time.time()
    aucs = []
    for train_idx, val_idx in folds:
        X_fit, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fit, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_fit, X_val = make_features(X_fit.copy(), X_val.copy())
        assert list(X_fit.columns) == list(X_val.columns), 'fit/apply columns differ'
        model = make_model()
        model.fit(X_fit, y_fit)
        aucs.append(float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])))
    return {'cv_auc': float(np.mean(aucs)), 'fold_aucs': aucs,
            'seconds': time.time() - start}
