"""Validation functions — confirm every data-quality correction is completed.

These are **validation checks**, not unit tests: each one runs against the real
corrected output of ``correct_data_issues`` (in ``correct_data_issues.py``) and
confirms that one of the data-quality issues from
``notebooks/02_data_quality_first_txn.ipynb`` is fully resolved. Together they
make sure all the updates are completed and no known issue survives in the data
that goes downstream.

Each ``validate_*`` function takes the corrected DataFrame and raises
``DataValidationError`` if the issue is still present. ``validate_all`` runs every
check, collects all failures, and raises a single ``DataValidationError`` listing
every check that did not pass (so a failing validation always surfaces as an
error, and you see all problems at once).

Run from anywhere:
    python src/data_validation_tests.py

which loads the first-transaction table, applies ``correct_data_issues`` and
validates the result.
"""
from pathlib import Path

import pandas as pd

from correct_data_issues import (
    INVALID_STOCK_CODES,
    IN_FILE,
    TEST_STOCK_CODE_PREFIX,
    correct_data_issues,
)


class DataValidationError(Exception):
    """Raised when a corrected-data validation check does not pass."""


def validate_no_zero_prices(clean: pd.DataFrame) -> None:
    """Issue 2 — raise unless no line item has ``price == 0``."""
    n = int((clean['price'] == 0).sum())
    if n:
        raise DataValidationError(f'{n} rows still have price == 0')


def validate_consistent_descriptions(clean: pd.DataFrame) -> None:
    """Issue 3 — raise unless every ``stock_code`` maps to a single ``description``."""
    n = int((clean.groupby('stock_code')['description'].nunique() > 1).sum())
    if n:
        raise DataValidationError(f'{n} stock codes still have more than one description')


def validate_no_invalid_stock_codes(clean: pd.DataFrame) -> None:
    """Issue 5 — raise unless none of the non-product stock codes remain."""
    n = int(clean['stock_code'].isin(INVALID_STOCK_CODES).sum())
    if n:
        raise DataValidationError(f'{n} rows still have an invalid stock code')


def validate_no_cancellations(clean: pd.DataFrame) -> None:
    """Issue 6 — raise unless no ``invoice`` starts with 'C' (cancellation)."""
    n = int(clean['invoice'].astype(str).str.startswith('C').sum())
    if n:
        raise DataValidationError(f'{n} cancellation rows still remain')


def validate_no_test_products(clean: pd.DataFrame) -> None:
    """Issue 8 — raise unless no ``TEST*`` placeholder stock code remains."""
    is_test = (clean['stock_code'].astype(str).str.upper()
               .str.startswith(TEST_STOCK_CODE_PREFIX.upper()))
    n = int(is_test.sum())
    if n:
        raise DataValidationError(f'{n} test-product rows still remain')


CHECKS = [
    validate_no_zero_prices,
    validate_consistent_descriptions,
    validate_no_invalid_stock_codes,
    validate_no_cancellations,
    validate_no_test_products,
]


def validate_all(clean: pd.DataFrame) -> None:
    """Run every validation check and raise if any of them fail.

    Runs all checks in ``CHECKS``, collects the ones that fail, and — if there is
    at least one failure — raises a single ``DataValidationError`` listing every
    check that did not pass. Returns ``None`` when all corrections are confirmed.
    """
    failures = []
    for check in CHECKS:
        try:
            check(clean)
        except DataValidationError as err:
            failures.append(str(err))

    if failures:
        raise DataValidationError(
            f'Data validation failed — {len(failures)} of {len(CHECKS)} checks '
            'did not pass:\n  - ' + '\n  - '.join(failures)
        )


def main() -> None:
    """Correct the first-transaction data and confirm all updates are completed."""
    df = pd.read_csv(IN_FILE, parse_dates=['invoice_date'])
    clean = correct_data_issues(df)

    validate_all(clean)
    print(f'All corrections validated on {len(clean):,} rows '
          f'({clean["customer_id"].nunique():,} customers) — no known issues remain.')


if __name__ == '__main__':
    main()
