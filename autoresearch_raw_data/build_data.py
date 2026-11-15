"""HUMAN-ONLY. Build the raw line-item train/test tables for the agent.

Writes, in ``data/processed/`` (git-ignored):

``raw_train.csv`` / ``raw_test.csv``
    One row per line item of each customer's FIRST invoice (the corrected rows from
    ``first_transaction_churn_clean.csv``), with these columns and nothing derived:
    ``invoice, stock_code, description, quantity, invoice_date, price, customer_id,
    country, churn``. ``churn`` is the per-customer label, repeated on each line.

The customers and the split are exactly those of ``src/split_data.py``: customers
are ordered by first purchase and the ones whose first purchase is before the
cutoff in ``results/split_summary.json`` are train, the rest test. The script
checks the sizes against that summary and that each customer's first purchase time
equals ``first_date`` in ``modelling_dataset.csv``.

    python autoresearch_raw_data/build_data.py
"""
import json
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
LINES_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'
MODELLING_FILE = PROCESSED_DIR / 'modelling_dataset.csv'
SPLIT_SUMMARY = PROJECT_ROOT / 'results' / 'split_summary.json'
TRAIN_FILE = PROCESSED_DIR / 'raw_train.csv'
TEST_FILE = PROCESSED_DIR / 'raw_test.csv'

COLUMNS = ['invoice', 'stock_code', 'description', 'quantity', 'invoice_date',
           'price', 'customer_id', 'country', 'churn']


def main() -> None:
    lines = pd.read_csv(LINES_FILE, parse_dates=['invoice_date'])
    summary = json.loads(SPLIT_SUMMARY.read_text())
    cutoff = pd.Timestamp(summary['cutoff'])

    first = lines.groupby('customer_id')['invoice_date'].min()
    reference = pd.read_csv(MODELLING_FILE, usecols=['customer_id', 'first_date'],
                            parse_dates=['first_date']).set_index('customer_id')['first_date']
    assert set(first.index) == set(reference.index), 'customer sets differ'
    assert (first.sort_index() == reference.sort_index()).all(), \
        'first purchase time differs from modelling_dataset.first_date'
    assert lines.groupby('customer_id')['churn'].nunique().max() == 1, 'churn varies within customer'

    is_train = lines['customer_id'].map(first) < cutoff
    train, test = lines[is_train][COLUMNS], lines[~is_train][COLUMNS]

    assert train['customer_id'].nunique() == summary['train']['rows'], 'train size differs'
    assert test['customer_id'].nunique() == summary['test']['rows'], 'test size differs'
    assert not set(train['customer_id']) & set(test['customer_id'])

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    print(f"train: {len(train):,} lines, {train['customer_id'].nunique():,} customers -> {TRAIN_FILE}")
    print(f"test : {len(test):,} lines, {test['customer_id'].nunique():,} customers -> {TEST_FILE}")


if __name__ == '__main__':
    main()
