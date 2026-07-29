"""Correct the data-quality issues found in ``notebooks/02_data_quality_first_txn.ipynb``.

Input is the first-transaction churn table produced by ``prepare_churn_data.py``
(``data/processed/first_transaction_churn.csv``): one row per line item of each
customer's **first** invoice, plus a per-customer ``churn`` label.

Each of the 7 issues from the EDA is handled per the notebook's "Resolution plan":

1. Missing ``customer_id`` / ``description`` — already resolved upstream by the
   first-transaction filter (no such rows remain); nothing to do here.
2. Zero prices (``price == 0``)          -> ``remove_zero_prices``.
3. One stock code -> many descriptions   -> ``normalize_descriptions`` (give every
   row of a ``stock_code`` its most common description).
4. One description -> many stock codes    — ``stock_code`` is treated as the source
   of truth and descriptions are left as they are; no separate step is needed.
5. Invalid stock-code formats            -> ``remove_invalid_stock_codes`` (drop the
   non-product codes in ``INVALID_STOCK_CODES``).
6. Cancellations (``C`` invoices)        -> ``remove_cancellations`` (a customer
   whose first transaction is a cancellation is dropped).
7. Outliers                               — kept on purpose: the downstream model is
   tree-based and robust to extreme values; no step is needed.

``correct_data_issues`` applies steps 2, 3, 5 and 6 in order and returns the
cleaned frame.

Run from anywhere:
    python src/correct_data_issues.py

Reads ``data/processed/first_transaction_churn.csv`` and writes
``data/processed/first_transaction_churn_clean.csv``.
"""
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
IN_FILE = PROCESSED_DIR / 'first_transaction_churn.csv'
OUT_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'

# Issue 5: letters-only / non-product stock codes that are not real products
# (postage, manual adjustments, bank charges, ...). Verified against the data
# in the EDA notebook — these 7 codes account for all invalid-format rows.
INVALID_STOCK_CODES = ['POST', 'M', 'D', 'ADJUST', 'PADS', 'DOT', 'BANK CHARGES']


def remove_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Issue 2 — drop line-item rows whose ``price`` is 0.

    Input:
        ``df``: first-transaction frame with a ``price`` column.
    Does:
        Keeps only rows where ``price != 0``.
    Output:
        A copy of ``df`` with the zero-price rows removed.
    """
    return df[df['price'] != 0].copy()


def normalize_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Issue 3 — give every row of a ``stock_code`` its most common ``description``.

    Input:
        ``df``: first-transaction frame with ``stock_code`` and ``description``.
    Does:
        For each ``stock_code`` picks the modal (most frequent) description and
        maps it onto all of that code's rows. The differences are mostly commas,
        synonyms and typos, so the modal label is the reliable one. Ties are
        broken alphabetically by ``Series.mode`` (deterministic).
    Output:
        A copy of ``df`` with ``description`` made consistent per ``stock_code``.
    """
    modal = df.groupby('stock_code')['description'].agg(lambda s: s.mode().iloc[0])
    out = df.copy()
    out['description'] = out['stock_code'].map(modal)
    return out


def remove_invalid_stock_codes(
    df: pd.DataFrame, invalid_codes: list[str] = INVALID_STOCK_CODES
) -> pd.DataFrame:
    """Issue 5 — drop rows whose ``stock_code`` is a non-product code.

    Input:
        ``df``: first-transaction frame with a ``stock_code`` column.
        ``invalid_codes``: codes to remove (defaults to ``INVALID_STOCK_CODES``).
    Does:
        Removes the individual line-item rows carrying an invalid code (e.g. a
        ``POST`` postage line inside an otherwise valid first invoice); the
        customer's real product lines are kept.
    Output:
        A copy of ``df`` without the invalid-code rows.
    """
    return df[~df['stock_code'].isin(invalid_codes)].copy()


def remove_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """Issue 6 — drop cancellation first transactions (``invoice`` starts with 'C').

    Input:
        ``df``: first-transaction frame with an ``invoice`` column.
    Does:
        In this table each customer has a single (first) invoice, so removing its
        cancellation rows removes that customer. A cancellation is not treated as
        a genuine first purchase because the real prior purchase it cancels is not
        present in the data.
    Output:
        A copy of ``df`` without cancellation rows.
    """
    is_cancel = df['invoice'].astype(str).str.startswith('C')
    return df[~is_cancel].copy()


def correct_data_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all corrections (issues 2, 3, 5, 6) and return the cleaned frame.

    Order: remove cancellations -> remove zero prices -> remove invalid stock
    codes -> normalize descriptions (computed on the surviving rows).

    Issues 1, 4 and 7 need no transformation (see the module docstring).
    """
    df = remove_cancellations(df)
    df = remove_zero_prices(df)
    df = remove_invalid_stock_codes(df)
    df = normalize_descriptions(df)
    return df


def main() -> None:
    """Load the first-transaction data, correct the issues and write the result.

    Reads ``IN_FILE``, applies ``correct_data_issues``, prints before/after row
    and customer counts, and writes ``OUT_FILE``.
    """
    df = pd.read_csv(IN_FILE, parse_dates=['invoice_date'])
    print(f'Loaded first-transaction data : {df.shape}')

    clean = correct_data_issues(df)
    print(f'After corrections             : {clean.shape}')
    print(f'Rows removed                  : {len(df) - len(clean):,}')
    print(f'Customers remaining           : {clean["customer_id"].nunique():,} '
          f'(was {df["customer_id"].nunique():,})')

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_FILE, index=False)
    print(f'Saved -> {OUT_FILE}')


if __name__ == '__main__':
    main()
