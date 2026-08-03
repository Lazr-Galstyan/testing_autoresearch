"""Build the two feature tables both pipelines start from.

Input for both is ``data/processed/first_transaction_churn_clean.csv`` — the
corrected line-item table from ``correct_data_issues.py`` (one row per line item
of each customer's first invoice). Two tables come out of it:

``data/processed/customer_features.csv``
    The **modelling table**: one row per ``customer_id``, the shared starting
    point for the traditional and the autoresearch pipelines.
``data/processed/item_characteristics.csv``
    A ``(stock_code, date)`` lookup of how each product had traded **strictly
    before** that date, joinable back onto any transaction.

Scope
-----
The project benchmarks *human* against *AI* feature engineering, so anything
built here is engineering that neither pipeline gets credit for. This module
therefore stays at the level of mechanical roll-ups and direct decompositions of
what the raw columns already contain: counts, sums, a mean, the calendar parts of
``invoice_date``, and a frequency floor on ``country``. Ratios, RFM constructs,
basket composition, price dispersion and encodings are left to the two pipelines
to build **on top of** these tables, each in their own directory.

The item table is the one deliberate exception to "mechanical only": it is a
point-in-time aggregation that is easy to get subtly wrong, so it is built and
validated once here rather than twice downstream. It is *not* joined onto the
modelling table — how to summarise a basket's worth of product history to the
customer grain is a modelling decision, and belongs to the pipelines.

The modelling table
===================

Columns produced
----------------
=======================  ====================================================
``customer_id``          Customer key (one row each).
``churn``                Label: 1 if no 2nd purchase within 90 days of the
                         1st, else 0. Carried through from
                         ``prepare_churn_data.py``; not recomputed here.
``first_date``           Timestamp of the first transaction.
``year``                 Calendar year of the first transaction.
``month``                Calendar month, 1-12.
``day_of_month``         Day of month, 1-31.
``weekday``              Day of week as an integer, 0 = Monday … 6 = Sunday.
``time_of_day``          Hour bucketed into ``night`` (00:00-06:59),
                         ``morning`` (07:00-11:59), ``afternoon``
                         (12:00-16:59) and ``evening`` (17:00-23:59). The
                         retailer only trades 07:00-20:59, so ``night`` is
                         empty in the current data; the bucket exists so the
                         mapping stays total if the source data changes.
``country``              Customer country, **grouped**: countries with more
                         than ``COUNTRY_MIN_CUSTOMERS`` customers keep their
                         name, every other country becomes ``Other``. The raw
                         40-level column is not carried through.
``n_lines``              Number of line items on the first invoice.
``n_distinct_products``  Distinct ``stock_code`` values on that invoice. Lower
                         than ``n_lines`` when a product appears on several
                         lines.
``total_quantity``       Sum of ``quantity`` over the invoice.
``total_spend``          Sum of ``line_total`` (= quantity × price) over the
                         invoice — the total price of the transaction.
``avg_unit_price``       Unweighted mean of the line-item ``price``. The
                         quantity-weighted alternative is derivable downstream
                         as ``total_spend / total_quantity``.
=======================  ====================================================

The label is **not** recomputed here: it depends on the customer's 2nd purchase,
which lives outside the first-transaction table entirely.

A note on ``country`` grouping and data leakage
-----------------------------------------------
The set of countries clearing ``COUNTRY_MIN_CUSTOMERS`` is computed over the
whole table, before any train/test split. The threshold never looks at ``churn``,
so this is an unsupervised grouping rather than target leakage — but the category
list is still informed by rows that later become test data. Recompute the list on
the training split alone if the benchmark needs to be airtight on this point.

The item-history table
======================

For every ``(stock_code, date)`` pair that appears in the corrected
first-transaction table, this summarises how that product had traded **strictly
before that date** — "how established was this product at the moment this
customer bought it?".

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

Run from anywhere:
    python src/build_features.py
"""
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
IN_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'
CUSTOMER_OUT_FILE = PROCESSED_DIR / 'customer_features.csv'
ITEM_OUT_FILE = PROCESSED_DIR / 'item_characteristics.csv'

# Countries with MORE than this many customers keep their own name; the rest are
# pooled into OTHER_COUNTRY. 04_eda_first_txn.ipynb §7 shows why: 26 of the 40
# countries hold fewer than 10 customers, so their per-country rates are
# arithmetic artefacts rather than behaviour.
COUNTRY_MIN_CUSTOMERS = 10
OTHER_COUNTRY = 'Other'

# Hour-of-day buckets. Edges are right-inclusive on the hour integer, and span
# all 24 hours so the mapping can never produce a missing value.
TIME_OF_DAY_BINS = [-1, 6, 11, 16, 23]
TIME_OF_DAY_LABELS = ['night', 'morning', 'afternoon', 'evening']

CUSTOMER_COLUMNS = [
    'customer_id',
    'churn',
    'first_date',
    'year',
    'month',
    'day_of_month',
    'weekday',
    'time_of_day',
    'country',
    'n_lines',
    'n_distinct_products',
    'total_quantity',
    'total_spend',
    'avg_unit_price',
]

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


# --------------------------------------------------------------------------
# Customer-level modelling table
# --------------------------------------------------------------------------

def group_countries(country: pd.Series,
                    min_customers: int = COUNTRY_MIN_CUSTOMERS) -> pd.Series:
    """Pool low-frequency countries into a single ``Other`` category.

    Input:
        ``country``: one country per customer (customer grain, not line grain —
        counting at line grain would weight by basket size).
        ``min_customers``: countries with strictly more than this many customers
        keep their own name.
    Does:
        Counts customers per country and replaces every country at or below the
        threshold with ``OTHER_COUNTRY``.
    Output:
        A Series of the same length with the pooled values.
    """
    counts = country.value_counts()
    keep = set(counts[counts > min_customers].index)
    return country.where(country.isin(keep), OTHER_COUNTRY)


def add_date_parts(features: pd.DataFrame) -> pd.DataFrame:
    """Decompose ``first_date`` into its calendar parts.

    Input:
        ``features``: frame with a datetime ``first_date`` column.
    Does:
        Adds ``year``, ``month``, ``day_of_month``, ``weekday`` (0 = Monday) and
        ``time_of_day`` (hour bucketed per ``TIME_OF_DAY_BINS``). ``first_date``
        itself is kept — it is needed for any time-ordered split.
    Output:
        The same frame with the five columns added.
    """
    dt = features['first_date']
    features['year'] = dt.dt.year
    features['month'] = dt.dt.month
    features['day_of_month'] = dt.dt.day
    features['weekday'] = dt.dt.dayofweek
    features['time_of_day'] = pd.cut(dt.dt.hour, bins=TIME_OF_DAY_BINS,
                                     labels=TIME_OF_DAY_LABELS).astype(str)
    return features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the corrected line-item frame to one row per customer.

    Input:
        ``df``: corrected first-transaction frame (line-item grain) with
        ``customer_id``, ``invoice_date``, ``country``, ``stock_code``,
        ``quantity``, ``price``, ``line_total`` and ``churn``.
    Does:
        Groups by ``customer_id`` and computes the roll-ups listed in the module
        docstring, then decomposes the invoice date and groups ``country``.
        ``churn``, ``country`` and the invoice timestamp are constant within a
        customer's single first invoice, so ``first`` / ``min`` simply carry them
        through rather than aggregating anything away.
    Output:
        A frame with one row per customer and the columns in
        ``CUSTOMER_COLUMNS``, sorted by ``customer_id``.
    """
    grouped = df.groupby('customer_id')

    features = pd.DataFrame({
        'churn': grouped['churn'].first(),
        'first_date': grouped['invoice_date'].min(),
        'country': grouped['country'].first(),
        'n_lines': grouped.size(),
        'n_distinct_products': grouped['stock_code'].nunique(),
        'total_quantity': grouped['quantity'].sum(),
        'total_spend': grouped['line_total'].sum(),
        'avg_unit_price': grouped['price'].mean(),
    })

    features = add_date_parts(features)
    features['country'] = group_countries(features['country'])

    return features.reset_index().sort_values('customer_id')[CUSTOMER_COLUMNS]


def check_modelling_table(features: pd.DataFrame) -> None:
    """Raise ``AssertionError`` unless the table is fit to model on.

    Confirms one row per customer, no missing values, a two-class label,
    non-negative totals (negatives would mean cancellation rows survived the
    corrections), and that the derived calendar and category columns are within
    their expected domains.
    """
    assert not features['customer_id'].duplicated().any(), 'customer_id is not unique'
    assert not features.isna().any().any(), 'modelling table has missing values'
    assert set(features['churn'].unique()) == {0, 1}, 'churn is not binary'
    for col in ['n_lines', 'n_distinct_products', 'total_quantity', 'total_spend']:
        assert (features[col] >= 0).all(), f'{col} has negative values'

    assert features['month'].between(1, 12).all(), 'month outside 1-12'
    assert features['day_of_month'].between(1, 31).all(), 'day_of_month outside 1-31'
    assert features['weekday'].between(0, 6).all(), 'weekday outside 0-6'
    assert set(features['time_of_day']).issubset(TIME_OF_DAY_LABELS), \
        'unexpected time_of_day value'

    small = features['country'].value_counts()
    small = small[small.index != OTHER_COUNTRY]
    assert (small > COUNTRY_MIN_CUSTOMERS).all(), \
        f'a named country has {COUNTRY_MIN_CUSTOMERS} or fewer customers'


# --------------------------------------------------------------------------
# Point-in-time item history
# --------------------------------------------------------------------------

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


def build_item_characteristics(df: pd.DataFrame) -> pd.DataFrame:
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


def check_item_characteristics(items: pd.DataFrame, df: pd.DataFrame) -> None:
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
        # Daily means first, then average across days — the definition in full.
        daily_price = earlier.groupby('date')['price'].mean()
        assert abs(row.prior_avg_price - daily_price.mean()) < 1e-9, \
            f'prior_avg_price mismatch for {row.stock_code} on {row.date}'
        assert abs(row.prior_median_daily_price - daily_price.median()) < 1e-9, \
            f'prior_median_daily_price mismatch for {row.stock_code} on {row.date}'


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    """Build both tables from the corrected data, validate them and write them out."""
    df = pd.read_csv(IN_FILE, parse_dates=['invoice_date'])
    print(f'Loaded corrected line items : {df.shape}')

    features = build_features(df)
    check_modelling_table(features)

    print()
    print('Customer modelling table')
    print('------------------------')
    print(f'Modelling table             : {features.shape}')
    print(f'Customers                   : {len(features):,}')
    print(f'Churn rate                  : {features["churn"].mean():.2%} '
          f'({int(features["churn"].sum()):,} churned / '
          f'{int((features["churn"] == 0).sum()):,} retained)')

    named = features['country'].nunique() - (features['country'] == OTHER_COUNTRY).any()
    print(f'Countries                   : {named} named + "{OTHER_COUNTRY}" '
          f'({(features["country"] == OTHER_COUNTRY).sum():,} customers pooled)')
    print(f'Time-of-day buckets         : '
          f'{features["time_of_day"].value_counts().to_dict()}')

    items = build_item_characteristics(df)
    check_item_characteristics(items, df)

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
    print(items[['prior_transactions', 'prior_units',
                 'prior_median_daily_transactions', 'prior_median_daily_units',
                 'prior_avg_price', 'prior_median_daily_price']]
          .describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(CUSTOMER_OUT_FILE, index=False)
    print(f'\nSaved -> {CUSTOMER_OUT_FILE}')
    items.to_csv(ITEM_OUT_FILE, index=False)
    print(f'Saved -> {ITEM_OUT_FILE}')


if __name__ == '__main__':
    main()
