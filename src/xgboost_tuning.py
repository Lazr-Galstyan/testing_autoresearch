"""Tune, fit and evaluate the conventional XGBoost churn model.

Everything the conventional notebook does lives here, so the notebook only calls
these functions and shows their results.

The data
--------
``data/processed/train.csv`` and ``test.csv`` come from ``split_data.py``. Each
holds ``churn`` (the label), ``first_date`` and the features. ``first_date`` is
used only to order the training rows for cross-validation and is never fed to
the model; ``load_split`` separates it from the features.

Cross-validation
----------------
Time-ordered, not shuffled. The training customers are ordered by ``first_date``
and ``time_ordered_folds`` makes one fold per entry of ``TRAIN_FRACTIONS``: fold
``0.6`` trains on the earliest 60% of customers and validates on the remaining,
latest 40%; ``0.7`` trains on the earliest 70% and validates on the latest 30%;
and so on to ``0.9``. The validation windows therefore overlap and all end at the
last training date, and every validation customer is later than every training
customer in the same fold. Customers sharing the cutoff timestamp stay on the
validation side, as in ``split_data.time_split``.

As in the main split, a training customer's label looks 90 days past their first
purchase, so the last training customers of each fold are labelled using weeks
that overlap the start of its validation window. No gap is left between them.

Tuning
------
Optuna's TPE sampler (a Bayesian method) maximises the mean validation AUC-ROC
over the folds. The test set is touched once, by ``evaluate``, after the final
model has been fitted on the whole training set with the best parameters.
"""
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

RANDOM_SEED = 42
N_TRIALS = 100
TRAIN_FRACTIONS = (0.6, 0.7, 0.8, 0.9)
THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
TRAIN_FILE = PROCESSED_DIR / 'train.csv'
TEST_FILE = PROCESSED_DIR / 'test.csv'
RESULTS_FILE = RESULTS_DIR / 'conventional_xgboost.json'

LABEL_COLUMN = 'churn'
DATE_COLUMN = 'first_date'

# Fixed for every trial and for the final model; only the searched parameters vary.
FIXED_PARAMS = {
    'eval_metric': 'auc',
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
}


def load_split(train_file: Path = TRAIN_FILE, test_file: Path = TEST_FILE):
    """Load the train and test sets and separate label, dates and features.

    Output:
        ``(X_train, y_train, dates_train, X_test, y_test)``. ``dates_train`` is
        the training ``first_date`` Series, row for row with ``X_train``; it is
        for ordering the CV folds and is not part of the features. The test
        dates are dropped from ``X_test`` the same way.
    """
    train = pd.read_csv(train_file, parse_dates=[DATE_COLUMN])
    test = pd.read_csv(test_file, parse_dates=[DATE_COLUMN])

    assert list(train.columns) == list(test.columns), 'train and test columns differ'
    drop = [LABEL_COLUMN, DATE_COLUMN]
    return (train.drop(columns=drop), train[LABEL_COLUMN], train[DATE_COLUMN],
            test.drop(columns=drop), test[LABEL_COLUMN])


def time_ordered_folds(dates: pd.Series, train_fractions=TRAIN_FRACTIONS):
    """Build expanding-window folds ordered by date.

    Input:
        ``dates``: one date per training row.
        ``train_fractions``: for each fold, the share of customers (earliest
        first) that forms its training part.
    Does:
        For each fraction, the cutoff is the date at that position in the sorted
        order. Training is every row strictly before the cutoff and validation is
        every row at or after it, so ties on the cutoff stay together and the
        realised shares can differ slightly from the requested ones.
    Output:
        A list of ``(train_positions, validation_positions)`` integer arrays,
        positions into ``dates``, one pair per fraction.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    ordered = dates.sort_values(kind='stable').reset_index(drop=True)
    folds = []
    for fraction in train_fractions:
        cutoff = ordered.iloc[int(round(len(ordered) * fraction))]
        folds.append((np.flatnonzero(dates < cutoff), np.flatnonzero(dates >= cutoff)))
    return folds


def check_folds(dates: pd.Series, folds) -> None:
    """Raise ``AssertionError`` unless every fold is disjoint and in time order."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    for k, (train_idx, val_idx) in enumerate(folds):
        assert len(train_idx) and len(val_idx), f'fold {k} has an empty part'
        assert not set(train_idx) & set(val_idx), f'fold {k}: train and validation overlap'
        assert len(train_idx) + len(val_idx) == len(dates), f'fold {k}: rows lost'
        assert dates.iloc[train_idx].max() < dates.iloc[val_idx].min(), \
            f'fold {k}: training is not strictly earlier than validation'
    train_sizes = [len(t) for t, _ in folds]
    assert train_sizes == sorted(train_sizes), 'folds are not expanding'


def suggest_params(trial: optuna.Trial) -> dict:
    """The search space: one value for each tuned XGBoost parameter."""
    return {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }


def fold_aucs(params: dict, X: pd.DataFrame, y: pd.Series, folds) -> list:
    """Validation AUC-ROC of ``params`` on each fold.

    Each fold fits a fresh model on its training rows and scores it on its
    validation rows, which are later in time.
    """
    aucs = []
    for train_idx, val_idx in folds:
        model = xgb.XGBClassifier(**params, **FIXED_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(float(roc_auc_score(y.iloc[val_idx], proba)))
    return aucs


def tune(X: pd.DataFrame, y: pd.Series, folds, n_trials: int = N_TRIALS,
         seed: int = RANDOM_SEED, show_progress_bar: bool = False) -> optuna.Study:
    """Search for the parameters with the highest mean validation AUC-ROC.

    Does:
        Runs ``n_trials`` Optuna trials with a seeded TPE sampler. A trial's
        value is the mean of ``fold_aucs`` over ``folds``, and its per-fold AUCs
        are stored on the trial as ``fold_aucs``.
    Output:
        The finished ``optuna.Study``.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        aucs = fold_aucs(suggest_params(trial), X, y, folds)
        trial.set_user_attr('fold_aucs', aucs)
        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


def fit_final(best_params: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit the final model on the whole training set with the tuned parameters."""
    model = xgb.XGBClassifier(**best_params, **FIXED_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = THRESHOLD) -> dict:
    """Score ``model`` on ``X`` and ``y``.

    Output:
        A dict with ``accuracy``, ``precision``, ``recall``, ``f1`` (all at
        ``threshold``), the threshold-free ``auc_roc``, and the confusion matrix
        counts ``tn``, ``fp``, ``fn``, ``tp``, with churn as the positive class.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': threshold,
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y, proba)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def save_results(study: optuna.Study, metrics: dict, X_train: pd.DataFrame,
                 X_test: pd.DataFrame, folds, path: Path = RESULTS_FILE) -> Path:
    """Write the metrics, best parameters and CV details to ``path`` as JSON."""
    output = {
        'approach': 'conventional_xgboost',
        'random_seed': RANDOM_SEED,
        'tuner': 'optuna TPE',
        'n_trials': len(study.trials),
        'cv': {
            'type': 'time-ordered expanding window',
            'train_fractions': list(TRAIN_FRACTIONS),
            'fold_train_sizes': [len(t) for t, _ in folds],
            'fold_validation_sizes': [len(v) for _, v in folds],
            'objective': 'mean validation AUC-ROC',
            'best_mean_auc': float(study.best_value),
            'best_fold_aucs': study.best_trial.user_attrs['fold_aucs'],
        },
        'best_params': study.best_params,
        'test_metrics': metrics,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X_train.shape[1],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + '\n')
    return path
