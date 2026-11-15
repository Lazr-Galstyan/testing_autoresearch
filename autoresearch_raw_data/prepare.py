"""Fixed evaluation harness for the autoresearch loop (approach 3: raw line items).

The agent MUST NOT edit this file. It owns everything that has to stay constant:

* the data: ``data/processed/raw_train.csv`` only (line items of each training
  customer's first invoice). The test set is never loaded here.
* the customers and CV folds: the same as the conventional and shared-features
  approaches. Customers are ordered by first purchase time and the folds are the
  expanding windows with train fractions 0.6/0.7/0.8/0.9 (see
  ``src/xgboost_tuning.time_ordered_folds``).
* the objective: mean validation AUC-ROC over those folds.

The agent's ``train.py`` supplies ``make_features`` and ``make_model``;
``run_cv`` calls them once per fold.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
TRAIN_FRACTIONS = (0.6, 0.7, 0.8, 0.9)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / 'data' / 'processed' / 'raw_train.csv'

CUSTOMER_COLUMN = 'customer_id'
DATE_COLUMN = 'invoice_date'
LABEL_COLUMN = 'churn'


def load_lines(path: Path = TRAIN_FILE) -> pd.DataFrame:
    """Line items with ``churn`` attached to each row."""
    return pd.read_csv(path, parse_dates=[DATE_COLUMN])


def customer_table(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: ``first_date`` (earliest line time) and ``churn``."""
    table = lines.groupby(CUSTOMER_COLUMN).agg(first_date=(DATE_COLUMN, 'min'),
                                               churn=(LABEL_COLUMN, 'first'))
    return table.sort_index()


def time_ordered_folds(dates: pd.Series, train_fractions=TRAIN_FRACTIONS):
    """Expanding-window folds; same logic as ``src/xgboost_tuning.time_ordered_folds``.

    Returns ``(train_positions, validation_positions)`` arrays, positions into
    ``dates``. Training is every row strictly before the cutoff date, validation
    every row at or after it.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    ordered = dates.sort_values(kind='stable').reset_index(drop=True)
    folds = []
    for fraction in train_fractions:
        cutoff = ordered.iloc[int(round(len(ordered) * fraction))]
        folds.append((np.flatnonzero(dates < cutoff), np.flatnonzero(dates >= cutoff)))
    return folds


def split_lines(lines: pd.DataFrame):
    """``(line items without the label, one-row-per-customer labels)``."""
    customers = customer_table(lines)
    return lines.drop(columns=[LABEL_COLUMN]), customers[LABEL_COLUMN]


def build_features(make_features, lines_fit, y_fit, lines_apply, apply_ids):
    """Call the agent's ``make_features`` and check what it returns.

    ``X_fit`` must have exactly one row per customer in ``y_fit`` and ``X_apply``
    exactly one per customer in ``apply_ids``, both indexed by ``customer_id``
    and with the same columns. Rows are put in the order of ``y_fit.index`` and
    ``apply_ids`` respectively.
    """
    X_fit, X_apply = make_features(lines_fit.copy(), y_fit.copy(), lines_apply.copy())
    for name, X, ids in (('X_fit', X_fit, y_fit.index), ('X_apply', X_apply, apply_ids)):
        assert X.index.is_unique, f'{name}: customer_id index has duplicates'
        assert set(X.index) == set(ids) and len(X) == len(ids), \
            f'{name}: must have exactly one row per customer'
    X_fit, X_apply = X_fit.loc[y_fit.index], X_apply.loc[apply_ids]
    assert list(X_fit.columns) == list(X_apply.columns), 'fit/apply columns differ'
    return X_fit, X_apply


def run_cv(make_features, make_model) -> dict:
    """Score the agent's pipeline with the fixed time-ordered CV.

    Input:
        ``make_features(lines_fit, y_fit, lines_apply) -> (X_fit, X_apply)``.
        ``lines_*`` are line-item DataFrames (no label column). ``y_fit`` is the
        churn label of the fit customers, a Series indexed by ``customer_id``.
        Return one row per customer, indexed by ``customer_id``. Anything learned
        from data must come from ``lines_fit`` / ``y_fit`` only. It is called per
        fold with that fold's training customers as fit and its validation
        customers as apply, so nothing from the validation window can leak in.
        ``make_model() -> estimator`` with ``fit(X, y)`` and ``predict_proba(X)``.
    Output:
        ``{'cv_auc': mean, 'fold_aucs': [...], 'seconds': wall clock}``.
    """
    lines, y = split_lines(load_lines())
    customers = customer_table(load_lines())
    folds = time_ordered_folds(customers['first_date'])

    start = time.time()
    aucs = []
    for fit_pos, val_pos in folds:
        fit_ids, val_ids = y.index[fit_pos], y.index[val_pos]
        X_fit, X_val = build_features(
            make_features,
            lines[lines[CUSTOMER_COLUMN].isin(fit_ids)], y.loc[fit_ids],
            lines[lines[CUSTOMER_COLUMN].isin(val_ids)], val_ids)
        model = make_model()
        model.fit(X_fit, y.loc[fit_ids])
        aucs.append(float(roc_auc_score(y.loc[val_ids], model.predict_proba(X_val)[:, 1])))
    return {'cv_auc': float(np.mean(aucs)), 'fold_aucs': aucs,
            'seconds': time.time() - start}
