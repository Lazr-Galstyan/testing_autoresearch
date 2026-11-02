"""Split the modelling dataset by time and reduce the embeddings with PCA.

Input is ``data/processed/modelling_dataset.csv`` (one row per customer, 384
unreduced ``emb_*`` columns). Output is, in ``data/processed/``:

``train.csv`` / ``test.csv``
    The same customer-grain, aggregate and ``pca_*`` columns in both. The 384
    ``emb_*`` columns are replaced by ``pca_*`` columns.
``pca.joblib``
    The PCA fitted on the training rows, so any later customer can be projected
    onto the same axes.

and ``results/split_summary.json`` with the cutoff, sizes and churn rates.

The split
---------
Customers are ordered by ``first_date`` and the earliest ``1 - TEST_FRACTION``
become train; the rest become test. The unit is the customer, so no customer
appears twice. Customers sharing the cutoff timestamp all go to the same side,
so train ends strictly before test begins and the realised test share can differ
from 20% by a handful of customers.

PCA
---
Fitted on the training customers' mean embeddings alone, then applied unchanged
to the test customers (``pca.transform``). The test set therefore never
influences the axes. The component count follows ``PCA_VARIANCE`` on the
training set, so it can differ from the 96 found on all customers.

Known limitation: a training customer's label looks 90 days past their first
purchase, so the last training labels are resolved using weeks that overlap the
start of the test period. Leaving a 90-day gap between the two sets removes that
overlap; this module does not.

Run from anywhere:
    python src/split_data.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from build_modelling_dataset import EMBEDDING_COLUMNS
from description_embeddings import PCA_VARIANCE, pca_column_names, reduce_embeddings

RANDOM_SEED = 42
TEST_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
IN_FILE = PROCESSED_DIR / 'modelling_dataset.csv'
TRAIN_FILE = PROCESSED_DIR / 'train.csv'
TEST_FILE = PROCESSED_DIR / 'test.csv'
PCA_FILE = PROCESSED_DIR / 'pca.joblib'
SUMMARY_FILE = RESULTS_DIR / 'split_summary.json'


def time_split(dataset: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split customers into an earlier train set and a later test set.

    Input:
        ``dataset``: the modelling table, with ``first_date`` per customer.
        ``test_fraction``: share of customers wanted in the test set.
    Does:
        Takes the ``first_date`` at the ``1 - test_fraction`` quantile position
        as the cutoff. Train is every customer strictly before it, test is every
        customer at or after it, so ties on the cutoff stay together.
    Output:
        ``(train, test, cutoff)`` — two frames in the original column order and
        the cutoff timestamp.
    """
    dates = pd.to_datetime(dataset['first_date'])
    ordered = dates.sort_values(kind='stable')
    cutoff = ordered.iloc[int(round(len(ordered) * (1 - test_fraction)))]
    train = dataset[dates < cutoff].reset_index(drop=True)
    test = dataset[dates >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def fit_and_apply_pca(train: pd.DataFrame, test: pd.DataFrame,
                      variance: float = PCA_VARIANCE):
    """Fit PCA on the training embedding means and project both sets.

    Does:
        ``reduce_embeddings`` is fitted on ``train`` only. ``test`` goes through
        ``pca.transform`` with those same components.
    Output:
        ``(train_out, test_out, pca)`` — the frames with ``emb_*`` replaced by
        ``pca_*`` (in the same position), and the fitted ``PCA``.
    """
    reduced_train, pca = reduce_embeddings(train[EMBEDDING_COLUMNS].to_numpy(),
                                           variance=variance)
    reduced_test = pca.transform(embedding_matrix(test)).astype(np.float32)
    names = pca_column_names(pca.n_components_)

    def swap(frame: pd.DataFrame, reduced: np.ndarray) -> pd.DataFrame:
        components = pd.DataFrame(reduced, columns=names, index=frame.index)
        return pd.concat([frame.drop(columns=EMBEDDING_COLUMNS), components], axis=1)

    return swap(train, reduced_train), swap(test, reduced_test), pca


def embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The embedding matrix of ``frame``, in the column order PCA was fitted on."""
    return frame[EMBEDDING_COLUMNS].to_numpy()


def check_split(dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                train_out: pd.DataFrame, test_out: pd.DataFrame, pca,
                cutoff: pd.Timestamp) -> None:
    """Raise ``AssertionError`` unless the split and the reduction are sound."""
    ids_train, ids_test = set(train['customer_id']), set(test['customer_id'])
    assert not ids_train & ids_test, 'a customer is in both train and test'
    assert len(ids_train) + len(ids_test) == len(dataset) == dataset['customer_id'].nunique(), \
        'the split does not partition the customers'
    assert pd.to_datetime(train['first_date']).max() < cutoff <= \
        pd.to_datetime(test['first_date']).min(), 'train does not end before test begins'

    assert list(train_out.columns) == list(test_out.columns), 'train/test columns differ'
    assert train_out.dtypes.equals(test_out.dtypes), 'train/test dtypes differ'

    # Nothing but the embedding columns may change: every other column must equal
    # the modelling table's value for the same customer.
    plain = [c for c in dataset.columns if c not in EMBEDDING_COLUMNS]
    original = dataset.set_index('customer_id')
    for part, out in ((train, train_out), (test, test_out)):
        pd.testing.assert_frame_equal(out[plain].set_index('customer_id'),
                                      original.loc[out['customer_id'], plain[1:]],
                                      check_dtype=True)
    assert not any(c in train_out.columns for c in EMBEDDING_COLUMNS), \
        'raw embedding columns survived the reduction'
    names = pca_column_names(pca.n_components_)
    assert [c for c in train_out.columns if c.startswith('pca_')] == names, \
        'pca columns are misnamed or out of order'

    # The components must come from train alone: refit and compare.
    from sklearn.decomposition import PCA
    refit = PCA(n_components=pca.n_components_, svd_solver='full',
                random_state=RANDOM_SEED).fit(embedding_matrix(train))
    assert np.allclose(np.abs(refit.components_), np.abs(pca.components_), atol=1e-6), \
        'the PCA axes do not match a training-only fit'
    assert np.allclose(pca.mean_, embedding_matrix(train).mean(axis=0), atol=1e-8), \
        'the PCA centre is not the training mean'

    dummies = [c for c in train_out.columns
               if c.startswith(('weekday_', 'time_of_day_', 'country_'))]
    assert dummies, 'no one-hot columns found'
    assert set(np.unique(np.concatenate([train_out[dummies].to_numpy().ravel(),
                                         test_out[dummies].to_numpy().ravel()]))) <= {0, 1}, \
        'a one-hot column is not 0/1'

    values = np.concatenate([train_out[names].to_numpy(), test_out[names].to_numpy()])
    assert np.isfinite(values).all(), 'a pca value is NaN or Inf'
    assert np.allclose(test_out[names].to_numpy(),
                       pca.transform(embedding_matrix(test)), atol=1e-4), \
        'test was not projected with the training PCA'


def main() -> None:
    """Split, reduce, validate and write the final train and test sets."""
    dataset = pd.read_csv(IN_FILE)
    print(f'Loaded modelling dataset : {dataset.shape}')

    train, test, cutoff = time_split(dataset)
    train_out, test_out, pca = fit_and_apply_pca(train, test)
    check_split(dataset, train, test, train_out, test_out, pca, cutoff)

    dummies = [c for c in train_out.columns
               if c.startswith(('weekday_', 'time_of_day_', 'country_'))]
    summary = {
        'random_seed': RANDOM_SEED,
        'split': 'time-ordered on first_date',
        'test_fraction_requested': TEST_FRACTION,
        'cutoff': str(cutoff),
        'train': {'rows': len(train_out),
                  'first_date_min': str(train['first_date'].min()),
                  'first_date_max': str(train['first_date'].max()),
                  'churn_rate': float(train['churn'].mean())},
        'test': {'rows': len(test_out),
                 'first_date_min': str(test['first_date'].min()),
                 'first_date_max': str(test['first_date'].max()),
                 'churn_rate': float(test['churn'].mean())},
        'one_hot_columns': len(dummies),
        'one_hot_all_zero_in_train': [c for c in dummies if train_out[c].sum() == 0],
        'one_hot_all_zero_in_test': [c for c in dummies if test_out[c].sum() == 0],
        'pca': {'fitted_on': 'train',
                'variance_target': PCA_VARIANCE,
                'n_components': int(pca.n_components_),
                'variance_explained': float(pca.explained_variance_ratio_.sum())},
    }

    print()
    print(f'Cutoff                   : {cutoff}')
    for name, part in (('train', summary['train']), ('test', summary['test'])):
        print(f'{name:<5}                    : {part["rows"]:,} customers, '
              f'{part["first_date_min"][:10]} -> {part["first_date_max"][:10]}, '
              f'churn {part["churn_rate"]:.2%}')
    print(f'PCA (fitted on train)    : {len(EMBEDDING_COLUMNS)} means -> '
          f'{pca.n_components_} components '
          f'({summary["pca"]["variance_explained"]:.1%} of variance)')
    print(f'One-hot columns          : {len(dummies)} (all-zero in train: '
          f'{summary["one_hot_all_zero_in_train"] or "none"}; in test: '
          f'{summary["one_hot_all_zero_in_test"] or "none"})')
    print(f'Final shape              : train {train_out.shape}, test {test_out.shape}')

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(TRAIN_FILE, index=False)
    test_out.to_csv(TEST_FILE, index=False)
    joblib.dump(pca, PCA_FILE)
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'\nSaved -> {TRAIN_FILE}\n         {TEST_FILE}\n         {PCA_FILE}\n         {SUMMARY_FILE}')


if __name__ == '__main__':
    main()
