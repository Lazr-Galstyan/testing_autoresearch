"""Assemble the final modelling dataset: one row per customer.

Input is ``data/processed/first_transaction_churn_clean.csv``. Output is:

``data/processed/modelling_dataset.csv``
    5,044 rows — one per customer — and 450 columns.

This module composes the other three rather than deriving anything new from the
raw line items::

    build_features.py          -> 16 customer-grain columns, joined as-is
    build_item_history.py      -> 10 line-varying metrics, aggregated 5 ways
    description_embeddings.py  -> 384-dim product vectors, averaged per customer

The order of operations
-----------------------
Everything that varies **within** a transaction is computed at line-item grain
first, then collapsed. Everything that is constant within a transaction is
never computed at line grain at all — it comes from ``build_features.py``,
which groups to one row per customer *before* deriving the calendar parts and
the country grouping.

That distinction is the whole design. Deriving ``country`` or ``weekday`` on
125,042 line items would produce the same value 25 times per customer and then
throw 24 of them away, and the frequency floor behind ``country`` counts
*customers* — computing it at line grain would weight every country by basket
size and change which countries clear the threshold.

So the flow is:

1. First transactions per customer, with the churn label  (upstream)
2. Line grain: join the item history, derive the four ratios, attach embeddings
3. Collapse to one row per customer by aggregating those line-varying metrics
4. Join the customer-grain columns — label, calendar parts, country, roll-ups

Columns produced
----------------
================================  ==========================================
16 from ``customer_features``     ``customer_id``, ``churn``, ``first_date``,
                                  the calendar parts, ``country`` /
                                  ``country_raw``, and the transaction
                                  roll-ups. Joined unchanged.
50 aggregates                     Each of ``LINE_METRIC_COLUMNS`` summarised
                                  by every entry in ``AGGREGATIONS``, named
                                  ``<column>_<aggregation>``.
384 embedding means               ``emb_000`` … ``emb_383``: the mean of the
                                  SBERT vectors of every product in the
                                  customer's basket.
================================  ==========================================

Why all five aggregations
-------------------------
Producing min, max, mean, median *and* sum for every metric is deliberately
exhaustive rather than selective. Choosing that a basket's product history is
best summarised by its maximum would be a feature-engineering decision, and
this project exists to compare how a human and an AI make exactly those
decisions — so making it here would hand both pipelines the same answer for
free. Generating all five and letting each pipeline select is mechanical.

Some combinations are close to meaningless on their own — the sum of
``prior_avg_price`` across a basket is not a price — and that is fine. They are
raw material for selection, not recommendations.

A note on summing missing values
--------------------------------
92 customers have no usable history on any line: every product in their basket
is making its first-ever appearance, so all four ratios are ``NaN`` for every
row they have. ``pandas`` sums an all-``NaN`` group to ``0.0`` by default, which
would give those customers a fabricated zero for ``_sum`` while every other
aggregation correctly reads ``NaN`` — and ``0`` is a meaningful value for these
ratios, so the fabrication would be invisible. ``aggregate_line_metrics``
therefore sums with ``min_count=1``, and all five aggregations agree on ``NaN``.

The 4% of *rows* with no history behave the same way: they drop out of their
customer's aggregation rather than counting as zeros, because every aggregation
here skips ``NaN``.

Run from anywhere:
    python src/build_modelling_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import build_features, check_modelling_table
from build_item_history import (RATIO_COLUMNS, add_history_ratios,
                                build_item_history, check_item_history)
from description_embeddings import (EMBEDDING_DIM, PCA_VARIANCE,
                                    build_description_embeddings, check_embeddings,
                                    load_model, pca_column_names, reduce_embeddings)

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
IN_FILE = PROCESSED_DIR / 'first_transaction_churn_clean.csv'
OUT_FILE = PROCESSED_DIR / 'modelling_dataset.csv'

AGGREGATIONS = ['min', 'max', 'mean', 'median', 'sum']

# The line-varying metrics that get collapsed to the customer. The six raw
# prior_* columns carry a product's absolute standing; the four ratios carry the
# current line relative to it. The ratios divide the absolute level out, so a
# niche product bought heavily and a bestseller bought lightly look alike in
# them — keeping both means that distinction survives to the pipelines.
PRIOR_COLUMNS = [
    'prior_transactions',
    'prior_units',
    'prior_median_daily_transactions',
    'prior_median_daily_units',
    'prior_avg_price',
    'prior_median_daily_price',
]
LINE_METRIC_COLUMNS = PRIOR_COLUMNS + RATIO_COLUMNS

EMBEDDING_PREFIX = 'emb_'
EMBEDDING_COLUMNS = [f'{EMBEDDING_PREFIX}{d:03d}' for d in range(EMBEDDING_DIM)]

# Aggregates are ordered by source column, so all five summaries of one metric
# sit together rather than all the minimums sitting together.
AGGREGATE_COLUMNS = [f'{col}_{agg}'
                     for col in LINE_METRIC_COLUMNS
                     for agg in AGGREGATIONS]


def build_line_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return the line-item frame carrying everything that varies within a basket.

    Input:
        ``df``: the corrected first-transaction frame (line-item grain).
    Does:
        Joins the point-in-time item history on ``(stock_code, date)`` and adds
        the four line-vs-history ratios. Deliberately adds **no** calendar or
        country columns: those are constant within a customer and come from
        ``build_features`` after the collapse.
    Output:
        ``df`` plus ``date``, the six ``prior_*`` columns and the four ratios.
    """
    items = build_item_history(df)
    check_item_history(items, df)

    txn = df.assign(date=df['invoice_date'].dt.normalize())
    txn = txn.merge(items, on=['stock_code', 'date'], how='left',
                    validate='many_to_one')
    assert len(txn) == len(df), 'the item-history merge changed the row count'
    assert txn['prior_transactions'].notna().all(), 'a line item found no history'

    return add_history_ratios(txn)


def aggregate_line_metrics(txn: pd.DataFrame) -> pd.DataFrame:
    """Collapse the line-varying metrics to one row per customer.

    Input:
        ``txn``: output of ``build_line_metrics``.
    Does:
        Summarises every column in ``LINE_METRIC_COLUMNS`` by every entry in
        ``AGGREGATIONS``. ``sum`` is computed separately with ``min_count=1``:
        pandas sums an all-``NaN`` group to ``0.0``, which would invent a zero
        for the 92 customers whose whole basket lacks history while the other
        four aggregations correctly report ``NaN``.
    Output:
        A frame indexed by ``customer_id`` with the ``AGGREGATE_COLUMNS``.
    """
    grouped = txn.groupby('customer_id')[LINE_METRIC_COLUMNS]

    parts = []
    for agg in AGGREGATIONS:
        part = (grouped.sum(min_count=1) if agg == 'sum'
                else grouped.agg(agg))
        part.columns = [f'{col}_{agg}' for col in part.columns]
        parts.append(part)

    return pd.concat(parts, axis=1)[AGGREGATE_COLUMNS]


def aggregate_embeddings(txn: pd.DataFrame, products: pd.DataFrame,
                         embeddings: np.ndarray) -> pd.DataFrame:
    """Average each customer's product vectors into one vector per customer.

    Input:
        ``txn``: any line-item frame with ``customer_id`` and ``stock_code``.
        ``products``, ``embeddings``: as returned by
        ``build_description_embeddings``.
    Does:
        Maps each line to its product's vector and takes the unweighted mean
        over the customer's line items — so a product appearing on two lines
        counts twice, which is what makes it a mean over the *basket* rather
        than over the distinct products in it.

        Accumulated one dimension at a time with ``bincount``. The direct route,
        indexing the matrix to line grain, would materialise a 125,042 x 384
        array (192 MB) purely to average it away; this holds one 125,042-element
        column at a time instead.
    Output:
        A frame indexed by ``customer_id`` with the 384 ``EMBEDDING_COLUMNS``.
    """
    position = pd.Index(products['stock_code']).get_indexer(txn['stock_code'])
    assert (position >= 0).all(), 'a line item has no product embedding'

    codes, customers = pd.factorize(txn['customer_id'], sort=True)
    lines_per_customer = np.bincount(codes, minlength=len(customers))

    sums = np.empty((len(customers), EMBEDDING_DIM), dtype=np.float64)
    for d in range(EMBEDDING_DIM):
        sums[:, d] = np.bincount(codes, weights=embeddings[position, d],
                                 minlength=len(customers))

    means = sums / lines_per_customer[:, None]
    return pd.DataFrame(means, index=pd.Index(customers, name='customer_id'),
                        columns=EMBEDDING_COLUMNS)


def reduce_customer_embeddings(embedding_means: pd.DataFrame,
                               variance: float = PCA_VARIANCE):
    """PCA the customer-level mean vectors down to the components that matter.

    Input:
        ``embedding_means``: the ``(n_customers, EMBEDDING_DIM)`` frame from
        ``aggregate_embeddings``, indexed by ``customer_id``.
        ``variance``: share of variance to keep.
    Does:
        Averaging first and reducing second means PCA is fitted on the customer
        means, which are far smoother than the product catalogue they came from
        — averaging roughly 18 unit vectors per basket concentrates them, so 90%
        of the variance survives in 96 components where the catalogue needs 140.
        The 384 raw means are intermediate and are not carried into the dataset.
    Output:
        ``(frame, pca)`` — a frame of ``pca_000`` … columns on the same index,
        and the fitted ``PCA``.
    """
    reduced, pca = reduce_embeddings(embedding_means.to_numpy(), variance=variance)
    frame = pd.DataFrame(reduced, index=embedding_means.index,
                         columns=pca_column_names(pca.n_components_))
    return frame, pca


def build_modelling_dataset(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Build the full modelling table from the corrected line items.

    Input:
        ``df``: the corrected first-transaction frame (line-item grain).
        ``model``: optional preloaded ``SentenceTransformer``.
    Does:
        Runs the four steps in the module docstring and joins the results on
        ``customer_id``.
    Output:
        One row per customer: the 16 ``customer_features`` columns, then the 50
        aggregates, then the 384 embedding means.
    """
    features = build_features(df)
    check_modelling_table(features)

    txn = build_line_metrics(df)
    aggregates = aggregate_line_metrics(txn)

    products, embeddings = build_description_embeddings(df, model=model)
    check_embeddings(products, embeddings)

    # Average to the customer first, then reduce — so the components are fitted
    # on the basket vectors that actually enter the model.
    embedding_means = aggregate_embeddings(txn, products, embeddings)
    embedding_components, _ = reduce_customer_embeddings(embedding_means)

    dataset = (features
               .merge(aggregates, on='customer_id', how='left', validate='one_to_one')
               .merge(embedding_components, on='customer_id', how='left',
                      validate='one_to_one'))
    return dataset


def check_modelling_dataset(dataset: pd.DataFrame, df: pd.DataFrame,
                            model=None) -> None:
    """Raise ``AssertionError`` unless the assembled table is correct.

    Confirms the grain and column set, that the label and customer list are
    unchanged from ``build_features``, that ``_sum`` reports ``NaN`` rather than
    ``0`` for a customer with no usable history, and — by recomputing a sample of
    customers directly from the line items — that the aggregations and the
    embedding means are what they claim to be.
    """
    features = build_features(df)

    assert not dataset['customer_id'].duplicated().any(), 'customer_id is not unique'
    assert len(dataset) == len(features), 'row count differs from customer_features'
    assert dataset['customer_id'].equals(features['customer_id']), \
        'customer list or order differs from customer_features'
    assert dataset['churn'].equals(features['churn']), 'churn changed'

    component_columns = [c for c in dataset.columns if c.startswith('pca_')]
    expected = list(features.columns) + AGGREGATE_COLUMNS + component_columns
    assert list(dataset.columns) == expected, 'unexpected column set or order'
    assert component_columns == pca_column_names(len(component_columns)), \
        'reduced dimensions are misnamed or out of order'

    reduced_values = dataset[component_columns].to_numpy()
    assert np.isfinite(reduced_values).all(), 'a reduced dimension is NaN or Inf'

    # An all-NaN basket must stay NaN in every aggregation, sum included.
    txn = build_line_metrics(df)
    blank = txn.groupby('customer_id')['qty_share_of_prior_units'].apply(
        lambda s: s.isna().all())
    blank = blank[blank].index
    if len(blank):
        rows = dataset.set_index('customer_id').loc[blank]
        for agg in AGGREGATIONS:
            col = f'qty_share_of_prior_units_{agg}'
            assert rows[col].isna().all(), \
                f'{col} is not NaN for a customer with no usable history'

    # Brute-force recomputation: the definitions, done the slow way.
    products, embeddings = build_description_embeddings(df, model=model)
    lookup = pd.Index(products['stock_code'])

    # The reduction, refit from scratch: deterministic given the same input, so
    # it must land on the same numbers the dataset carries.
    means = aggregate_embeddings(txn, products, embeddings)
    components, _ = reduce_customer_embeddings(means)
    assert np.abs(components.to_numpy() - reduced_values).max() < 1e-4, \
        'the reduced dimensions do not reproduce'
    sample = dataset['customer_id'].sample(min(50, len(dataset)),
                                           random_state=RANDOM_SEED)
    indexed = dataset.set_index('customer_id')
    for customer in sample:
        basket = txn[txn['customer_id'] == customer]
        row = indexed.loc[customer]

        for col in LINE_METRIC_COLUMNS:
            values = basket[col]
            for agg in AGGREGATIONS:
                want = (values.sum(min_count=1) if agg == 'sum'
                        else getattr(values, agg)())
                got = row[f'{col}_{agg}']
                assert (pd.isna(want) and pd.isna(got)) or abs(want - got) < 1e-6, \
                    f'{col}_{agg} mismatch for customer {customer}'

        # Hand-check the mean the reduction was applied to, not its projection.
        want_vector = embeddings[lookup.get_indexer(basket['stock_code'])].mean(axis=0)
        got_vector = means.loc[customer].to_numpy(dtype=float)
        assert np.abs(want_vector - got_vector).max() < 1e-6, \
            f'embedding mean mismatch for customer {customer}'


def main() -> None:
    """Assemble the modelling dataset, validate it and write it out."""
    df = pd.read_csv(IN_FILE, parse_dates=['invoice_date'])
    print(f'Loaded corrected line items : {df.shape}')

    model = load_model()
    dataset = build_modelling_dataset(df, model=model)
    check_modelling_dataset(dataset, df, model=model)

    print()
    print('Modelling dataset')
    print('-----------------')
    print(f'Shape                       : {dataset.shape}')
    print(f'Customers                   : {len(dataset):,}')
    print(f'Churn rate                  : {dataset["churn"].mean():.2%}')
    n_pca = sum(c.startswith('pca_') for c in dataset.columns)
    print(f'Columns                     : '
          f'{dataset.shape[1] - len(AGGREGATE_COLUMNS) - n_pca} customer-grain '
          f'+ {len(AGGREGATE_COLUMNS)} aggregates + {n_pca} reduced dimensions')
    print(f'Embedding reduction         : {EMBEDDING_DIM} means -> {n_pca} components '
          f'({PCA_VARIANCE:.0%} of customer-mean variance)')

    blank = dataset[f'{RATIO_COLUMNS[1]}_sum'].isna().sum()
    print(f'No usable product history   : {blank:,} customers '
          f'({blank / len(dataset) * 100:.1f}%) — NaN across all five aggregations')

    print()
    print('Aggregates (a sample of the 50):')
    sample_columns = [f'{c}_{a}' for c in ('prior_units', 'price_vs_prior_avg_price')
                      for a in AGGREGATIONS]
    print(dataset[sample_columns].describe(percentiles=[0.5]).round(2).to_string())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUT_FILE, index=False)
    print(f'\nSaved -> {OUT_FILE} '
          f'({OUT_FILE.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
