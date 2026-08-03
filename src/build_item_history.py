"""Build the point-in-time item-history lookup.

Input is ``data/processed/first_transaction_churn_clean.csv`` — the corrected
line-item table from ``correct_data_issues.py`` (one row per line item of each
customer's first invoice). Output is:

``data/processed/item_history.csv``
    A ``(stock_code, date)`` lookup of how each product had traded **strictly
    before** that date, joinable back onto any transaction.

Why this is its own module
--------------------------
This is a **lookup table, not a feature table**. ``build_features.py`` produces
one row per ``customer_id`` — the modelling grain, the unit of prediction — so
everything in it is a feature by construction. The table here sits at
``(stock_code, date)`` grain and is not a feature of anything until it is
aggregated up to a customer, and *how* to summarise a basket's worth of product
history to one customer row is a modelling decision that belongs to the two
pipelines. The join is a consequence of that grain mismatch.

It is built here rather than twice downstream for one reason: point-in-time
aggregation is easy to get subtly wrong, and getting it wrong leaks the future
into the features silently. Building and validating it once, in one place, means
both pipelines inherit the same verified history.

What it summarises
------------------
For every ``(stock_code, date)`` pair that appears in the corrected
first-transaction table, this answers "how established was this product at the
moment this customer bought it?".

The point-in-time rule
----------------------
A row dated 2010-03-03 is computed from line items dated **2010-03-02 and
earlier**. The current date is excluded, so nothing a customer could not have
"known" leaks into their own row.

Because the history is aggregated to whole days before any cumulative sum is
taken, **transactions sharing a date automatically share a cutoff**: two
customers buying the same product on 2010-03-03 both see history up to
2010-03-02, regardless of the time of day on their invoices. Time-of-day is
deliberately ignored.

Source scope
------------
History is drawn from the **first-transaction table only**, not the full raw
file. Every transaction here is some customer's first purchase, so these metrics
describe popularity **among first-time buyers** rather than the retailer's total
trade. The narrower definition keeps the feature inside the modelling dataset.

Metrics per ``(stock_code, date)``
----------------------------------
=====================================  ======================================
``prior_transactions``                 Total transactions containing the code
                                       before this date. A transaction is a
                                       unique ``invoice``.
``prior_units``                        Total units (sum of ``quantity``) of the
                                       code before this date.
``prior_median_daily_transactions``    Median, across the code's earlier
                                       **trading days**, of the transactions
                                       per day containing it.
``prior_median_daily_units``           Median, across the code's earlier
                                       **trading days**, of units per day.
``prior_avg_price``                    Mean, across the code's earlier trading
                                       days, of that day's average unit price.
``prior_median_daily_price``           Median, across the same days, of that
                                       day's average unit price.
=====================================  ======================================

Medians are taken over the days the product actually traded. Days on which the
code sold nothing are not included as zeros — with 4,171 codes over 648 calendar
days, zero-filling would drive nearly every median to 0 and destroy the signal.

The two price columns describe the **level** a product normally sold at, not
revenue. Both are built from daily average prices rather than from raw line
items, matching how the unit metrics work: a day on which the product sold to
forty customers counts once, exactly as a day with one customer does. Averaging
line items directly would instead let busy days dominate the historical price.
``04_eda_first_txn.ipynb`` §8a is the reason these are worth having — 1,681
codes (40.3%) sell at more than one price, so "what did this product usually
cost?" is a real question with a moving answer.

Cold start
----------
The first date a code ever appears has no prior history. Totals are ``0`` there
(genuinely nothing sold yet); the medians are ``NaN`` (undefined — there are no
earlier days to take a median over). Roughly 4% of pairs are in this state.
Callers should decide how to fill them; tree models handle the ``NaN`` directly.

A note on joining this back
---------------------------
``date`` is a whole day, so the join key on the transaction side is
``invoice_date.dt.normalize()``, not the raw timestamp. The result is
many-to-one: several line items of the same invoice can share a product-day.
``notebooks/05_feat_eng_data_split.ipynb`` does the join and asserts both.

Run from anywhere:
    python src/build_item_history.py
"""
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
IN_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'
OUT_FILE = PROCESSED_DIR / 'item_history.csv'

ITEM_COLUMNS = [
    'stock_code',
    'date',
    'prior_transactions',
    'prior_units',
    'prior_median_daily_transactions',
    'prior_median_daily_units',
    'prior_avg_price',
    'prior_median_daily_price',
]


def daily_item_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per ``(stock_code, date)``.

    Input:
        ``df``: first-transaction frame with ``stock_code``, ``invoice``,
        ``invoice_date``, ``quantity`` and ``price``.
    Does:
        Truncates ``invoice_date`` to the day and computes, for each product and
        day, how many distinct transactions contained it, how many units moved,
        and the mean unit price it sold at. Collapsing to the day here is what
        makes same-day transactions share a cutoff downstream, and what stops
        busy days dominating the price history.
    Output:
        A frame with ``stock_code``, ``date``, ``transactions``, ``units`` and
        ``avg_price``, sorted by product then date.
    """
    return (df.assign(date=df['invoice_date'].dt.normalize())
              .groupby(['stock_code', 'date'], as_index=False)
              .agg(transactions=('invoice', 'nunique'),
                   units=('quantity', 'sum'),
                   avg_price=('price', 'mean'))
              .sort_values(['stock_code', 'date'])
              .reset_index(drop=True))


def add_prior_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Add the strictly-prior totals and medians to a daily activity frame.

    Input:
        ``daily``: output of ``daily_item_activity``, sorted by product and date.
    Does:
        Within each ``stock_code``, accumulates over earlier dates only.
        Totals use ``cumsum`` minus the current day, which is exactly the sum of
        all earlier days and is ``0`` on a product's first date. Medians and the
        price mean use expanding windows shifted by one row, so the current day
        is excluded and the first date is ``NaN``.
    Output:
        ``daily`` with the six ``prior_*`` columns added.
    """
    grouped = daily.groupby('stock_code')

    daily['prior_transactions'] = grouped['transactions'].cumsum() - daily['transactions']
    daily['prior_units'] = grouped['units'].cumsum() - daily['units']

    daily['prior_median_daily_transactions'] = grouped['transactions'].transform(
        lambda s: s.expanding().median().shift(1))
    daily['prior_median_daily_units'] = grouped['units'].transform(
        lambda s: s.expanding().median().shift(1))

    # Price level: built from daily average prices so every trading day weighs
    # the same, however many customers bought that day.
    daily['prior_avg_price'] = grouped['avg_price'].transform(
        lambda s: s.expanding().mean().shift(1))
    daily['prior_median_daily_price'] = grouped['avg_price'].transform(
        lambda s: s.expanding().median().shift(1))

    return daily


def build_item_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return the ``(stock_code, date)`` lookup table of prior trading history.

    Input:
        ``df``: the corrected first-transaction frame (line-item grain).
    Does:
        Aggregates to product-days, then accumulates strictly-earlier history
        per product. See the module docstring for the metric definitions.
    Output:
        A frame with the columns in ``ITEM_COLUMNS``, one row per
        ``(stock_code, date)`` observed in the input.
    """
    daily = add_prior_metrics(daily_item_activity(df))
    return daily[ITEM_COLUMNS]


def check_item_history(items: pd.DataFrame, df: pd.DataFrame) -> None:
    """Raise ``AssertionError`` unless the lookup table is point-in-time correct.

    Verifies uniqueness of the key, that totals are non-negative, that every
    product's first date carries no prior history, and — by recomputing a sample
    of rows directly from the source with a plain date filter — that the
    cumulative logic matches a brute-force answer.
    """
    assert not items.duplicated(['stock_code', 'date']).any(), \
        '(stock_code, date) is not unique'
    assert (items['prior_transactions'] >= 0).all(), 'negative prior_transactions'
    assert (items['prior_units'] >= 0).all(), 'negative prior_units'
    assert (items['prior_avg_price'].dropna() > 0).all(), 'non-positive prior_avg_price'

    first_dates = items.groupby('stock_code')['date'].transform('min')
    opening = items[items['date'] == first_dates]
    assert (opening['prior_transactions'] == 0).all(), 'first date has prior transactions'
    assert opening['prior_median_daily_transactions'].isna().all(), \
        'first date has a prior median'
    assert opening['prior_avg_price'].isna().all(), 'first date has a prior price'

    # Brute-force recomputation on a sample: the definition, done the slow way.
    source = df.assign(date=df['invoice_date'].dt.normalize())
    sample = items.sample(min(200, len(items)), random_state=RANDOM_SEED)
    for row in sample.itertuples():
        earlier = source[(source['stock_code'] == row.stock_code)
                         & (source['date'] < row.date)]
        assert row.prior_transactions == earlier['invoice'].nunique(), \
            f'prior_transactions mismatch for {row.stock_code} on {row.date}'
        assert row.prior_units == earlier['quantity'].sum(), \
            f'prior_units mismatch for {row.stock_code} on {row.date}'

        if earlier.empty:
            assert pd.isna(row.prior_avg_price), \
                f'{row.stock_code} on {row.date} has a price with no history'
            continue

        daily = earlier.groupby('date').agg(t=('invoice', 'nunique'),
                                            u=('quantity', 'sum'),
                                            p=('price', 'mean'))
        assert row.prior_median_daily_transactions == daily['t'].median(), \
            f'prior_median_daily_transactions mismatch for {row.stock_code} on {row.date}'
        assert row.prior_median_daily_units == daily['u'].median(), \
            f'prior_median_daily_units mismatch for {row.stock_code} on {row.date}'
        # Daily means first, then average across days — the definition in full.
        assert abs(row.prior_avg_price - daily['p'].mean()) < 1e-9, \
            f'prior_avg_price mismatch for {row.stock_code} on {row.date}'
        assert abs(row.prior_median_daily_price - daily['p'].median()) < 1e-9, \
            f'prior_median_daily_price mismatch for {row.stock_code} on {row.date}'


def main() -> None:
    """Build the item-history table, validate it and write it out."""
    df = pd.read_csv(IN_FILE, parse_dates=['invoice_date'])
    print(f'Loaded corrected line items : {df.shape}')

    items = build_item_history(df)
    check_item_history(items, df)

    cold = int((items['prior_transactions'] == 0).sum())
    print()
    print('Item history table')
    print('------------------')
    print(f'(stock_code, date) rows     : {len(items):,}')
    print(f'Distinct stock codes        : {items["stock_code"].nunique():,}')
    print(f'Date range                  : {items["date"].min():%Y-%m-%d} -> '
          f'{items["date"].max():%Y-%m-%d}')
    print(f'Rows with no prior history  : {cold:,} ({cold / len(items) * 100:.1f}%)')
    print()
    print('Prior-history metrics:')
    print(items[[c for c in ITEM_COLUMNS if c.startswith('prior_')]]
          .describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    items.to_csv(OUT_FILE, index=False)
    print(f'\nSaved -> {OUT_FILE}')


if __name__ == '__main__':
    main()
