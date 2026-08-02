"""Aggregate the corrected first-transaction line items to one row per customer.

This is the **modelling table**: the shared starting point for both the
traditional and the autoresearch pipelines.

Input is ``data/processed/first_transaction_churn_clean.csv`` — the corrected
line-item table from ``correct_data_issues.py`` (one row per line item of each
customer's first invoice). Output is
``data/processed/customer_features.csv``, one row per ``customer_id``.

Scope
-----
The project benchmarks *human* against *AI* feature engineering, so anything
built here is engineering that neither pipeline gets credit for. This module
therefore stays at the level of mechanical roll-ups and direct decompositions of
what the raw columns already contain: counts, sums, a mean, the calendar parts of
``invoice_date``, and a frequency floor on ``country``. Ratios, RFM constructs,
basket composition, price dispersion and encodings are left to the two pipelines
to build **on top of** this table, each in their own directory.

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

Run from anywhere:
    python src/build_features.py
"""
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
IN_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'
OUT_FILE = PROCESSED_DIR / 'customer_features.csv'

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

COLUMNS = [
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
        A frame with one row per customer and the columns in ``COLUMNS``,
        sorted by ``customer_id``.
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

    return features.reset_index().sort_values('customer_id')[COLUMNS]


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


def main() -> None:
    """Build the modelling table from the corrected data and write it out."""
    df = pd.read_csv(IN_FILE, parse_dates=['invoice_date'])
    print(f'Loaded corrected line items : {df.shape}')

    features = build_features(df)
    check_modelling_table(features)

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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FILE, index=False)
    print(f'Saved -> {OUT_FILE}')


if __name__ == '__main__':
    main()
