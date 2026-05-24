"""Build customer-level features from each customer's 1st purchase.

Run from anywhere:
    python conventional_approach/feature_engineering.py

Writes data/processed/customers.csv.

Definition:
- One row per customer, taken from their 1st purchase.
- churn = 1 if there is no 2nd purchase, or the 2nd purchase is more than
  365 days after the 1st. churn = 0 if the 2nd purchase is within 365 days.
- Customers whose 1st purchase is within the last 365 days of the dataset
  are excluded (their churn label cannot be determined with full visibility).

Feature groups (also exported as module constants):
- ID: customer_id (index), customer_name.
- Categorical: product_category, payment_method, gender, purchase_weekday,
  purchase_month, purchase_time_of_day, is_holiday.
- Numeric: product_price, quantity, transaction_amount, customer_age,
  returns, purchase_year, purchase_hour, churn (target), plus the
  week-over-week % diff columns described below.

Notes:
- ``total_purchase_amount`` is dropped because it is not known in advance.
  ``transaction_amount = product_price * quantity`` is computed instead.
- ``is_holiday`` uses pandas' built-in ``USFederalHolidayCalendar``
  (equivalent to the US federal holiday set in the ``holidays`` package).
- Week-over-week % diff sums all customers' 1st transactions in
  [most recent Sunday 00:00, t] and compares with the same span shifted
  back 7 days. Null when the prior week is empty or precedes dataset start.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

RANDOM_SEED = 42
CHURN_WINDOW_DAYS = 365

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_FILE = RAW_DIR / 'ecommerce_customer_data_large.csv'
OUT_FILE = PROCESSED_DIR / 'customers.csv'

ID_COLS = ['customer_name']  # customer_id is the index
CATEGORICAL_COLS = ['product_category', 'payment_method', 'gender',
                    'purchase_weekday', 'purchase_month', 'purchase_time_of_day',
                    'is_holiday']


def time_of_day(hour: int) -> str:
    """Bucket an hour-of-day (0-23) into a named time-of-day label.

    Input:
        ``hour``: integer hour, 0-23.

    Does:
        Applies the bucket boundaries:
        06-12 -> Morning, 12-18 -> Afternoon, 18-22 -> Evening,
        22-06 (wrapping past midnight) -> Night.

    Output:
        One of 'Morning', 'Afternoon', 'Evening', 'Night'.
    """
    if 6 <= hour < 12:
        return 'Morning'
    if 12 <= hour < 18:
        return 'Afternoon'
    if 18 <= hour < 22:
        return 'Evening'
    return 'Night'


def load_raw() -> pd.DataFrame:
    """Load the raw transaction CSV from disk.

    Input:
        None. Reads ``RAW_FILE`` (``data/raw/ecommerce_customer_data_large.csv``).

    Does:
        - Reads the CSV into a DataFrame.
        - Normalises column names to lowercase with underscores.
        - Parses ``purchase_date`` from string into a pandas datetime.

    Output:
        ``pd.DataFrame`` with one row per raw transaction.
    """
    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    return df


def _compute_weekly_pct_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Week-over-week % difference of total quantity and total amount.

    Input:
        ``df``: DataFrame containing ``purchase_date``, ``quantity`` and
        ``transaction_amount`` columns.

    Does:
        For each row at timestamp ``t``:
          - ``current_period`` = [most recent Sunday 00:00, t]
          - ``prev_period``    = [Sunday of the previous week 00:00, t - 7 days]
        Sums quantity and transaction_amount across all rows in the dataset
        whose ``purchase_date`` falls in those windows, then computes the
        percent difference ``(current - prev) / prev * 100`` for each.
        Returns NaN when the previous-week sum is zero or when the previous
        week starts before the dataset's earliest date.

    Output:
        ``pd.DataFrame`` aligned to ``df.index`` with two columns:
        ``pct_diff_quantity_vs_prev_week`` and ``pct_diff_amount_vs_prev_week``.
    """
    df_sorted = df.sort_values('purchase_date')
    ts = df_sorted['purchase_date']
    qty = df_sorted['quantity'].astype(float).to_numpy()
    amt = df_sorted['transaction_amount'].astype(float).to_numpy()
    sorted_dates = ts.to_numpy()

    cumsum_qty = np.concatenate([[0.0], np.cumsum(qty)])
    cumsum_amt = np.concatenate([[0.0], np.cumsum(amt)])

    days_since_sun = (ts.dt.weekday + 1) % 7  # Mon=0..Sun=6 -> Sun=0, Mon=1, ...
    current_week_start = ts.dt.normalize() - pd.to_timedelta(days_since_sun, unit='D')
    prev_week_start = current_week_start - pd.Timedelta(days=7)
    prev_t = ts - pd.Timedelta(days=7)

    def range_sum(cumsum, start_series, end_series):
        i_lo = np.searchsorted(sorted_dates, start_series.to_numpy(), side='left')
        i_hi = np.searchsorted(sorted_dates, end_series.to_numpy(), side='right')
        return cumsum[i_hi] - cumsum[i_lo]

    cur_qty = range_sum(cumsum_qty, current_week_start, ts)
    prev_qty = range_sum(cumsum_qty, prev_week_start, prev_t)
    cur_amt = range_sum(cumsum_amt, current_week_start, ts)
    prev_amt = range_sum(cumsum_amt, prev_week_start, prev_t)

    dataset_min = pd.Timestamp(sorted_dates.min())
    invalid = (prev_week_start < dataset_min).to_numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        pct_qty = np.where(prev_qty > 0, (cur_qty - prev_qty) / prev_qty * 100, np.nan)
        pct_amt = np.where(prev_amt > 0, (cur_amt - prev_amt) / prev_amt * 100, np.nan)
    pct_qty[invalid] = np.nan
    pct_amt[invalid] = np.nan

    result = pd.DataFrame({
        'pct_diff_quantity_vs_prev_week': pct_qty,
        'pct_diff_amount_vs_prev_week': pct_amt,
    }, index=df_sorted.index)
    return result.reindex(df.index)


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw transactions to one row per customer with the engineered features.

    Input:
        ``df``: raw transaction-level ``pd.DataFrame`` as produced by
        ``load_raw``. Must contain ``customer_id``, ``customer_name``,
        ``purchase_date``, ``product_price``, ``quantity``, ``payment_method``,
        ``product_category``, ``gender``, ``customer_age``, ``returns``,
        ``total_purchase_amount`` and ``churn``.

    Does:
        - Sorts transactions by ``customer_id`` then ``purchase_date`` and
          ranks them per customer.
        - Keeps only each customer's 1st transaction and joins their 2nd
          transaction date.
        - Excludes customers whose 1st purchase is within
          ``CHURN_WINDOW_DAYS`` of the dataset end.
        - Assigns ``churn = 1`` if the customer has no 2nd purchase or the
          gap exceeds ``CHURN_WINDOW_DAYS`` days; ``churn = 0`` otherwise.
        - Drops ``total_purchase_amount`` (not known at acquisition) and
          replaces it with ``transaction_amount = product_price * quantity``.
        - Extracts ``purchase_weekday``, ``purchase_month`` (names),
          ``purchase_year``, ``purchase_hour`` and ``purchase_time_of_day``.
        - Flags ``is_holiday`` against the US federal holiday calendar.
        - Drops duplicates / helper columns (``age``, ``txn_rank``,
          ``second_purchase_date``).
        - Computes dataset-wide week-over-week % diff for total quantity
          and total amount.
        - Casts categorical columns to ``category`` dtype.

    Output:
        ``pd.DataFrame`` indexed by ``customer_id``, one row per eligible
        customer, containing the ID column, categorical columns, numeric
        columns and week-over-week % diff columns.
    """
    df_sorted = df.sort_values(['customer_id', 'purchase_date']).copy()
    df_sorted['txn_rank'] = df_sorted.groupby('customer_id').cumcount() + 1

    first = df_sorted[df_sorted['txn_rank'] == 1].set_index('customer_id').copy()
    second_date = (
        df_sorted.loc[df_sorted['txn_rank'] == 2, ['customer_id', 'purchase_date']]
        .set_index('customer_id')['purchase_date']
        .rename('second_purchase_date')
    )

    dataset_end = df['purchase_date'].max()
    cutoff = dataset_end - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    eligible = first[first['purchase_date'] <= cutoff].copy()
    eligible = eligible.join(second_date)

    gap_days = (
        (eligible['second_purchase_date'] - eligible['purchase_date'])
        .dt.total_seconds() / 86400
    )
    eligible['churn'] = (~(gap_days <= CHURN_WINDOW_DAYS)).astype(int)

    eligible = eligible.drop(columns=['total_purchase_amount'])
    eligible['transaction_amount'] = eligible['product_price'] * eligible['quantity']

    eligible['purchase_weekday']     = eligible['purchase_date'].dt.day_name()
    eligible['purchase_month']       = eligible['purchase_date'].dt.month_name()
    eligible['purchase_year']        = eligible['purchase_date'].dt.year
    eligible['purchase_hour']        = eligible['purchase_date'].dt.hour
    eligible['purchase_time_of_day'] = eligible['purchase_hour'].apply(time_of_day)

    cal = USFederalHolidayCalendar()
    holiday_set = set(
        cal.holidays(start=eligible['purchase_date'].min(),
                     end=eligible['purchase_date'].max()).date
    )
    eligible['is_holiday'] = eligible['purchase_date'].dt.date.isin(holiday_set)

    eligible = eligible.drop(columns=['txn_rank', 'second_purchase_date', 'age'])

    wow = _compute_weekly_pct_diff(eligible)
    eligible = eligible.join(wow)

    for col in CATEGORICAL_COLS:
        if col in eligible.columns:
            eligible[col] = eligible[col].astype('category')

    return eligible


def main() -> None:
    """Pipeline entry point: load raw data, build features and write to disk.

    Input:
        None. Reads paths from the module-level constants ``RAW_FILE`` and
        ``OUT_FILE``.

    Does:
        - Calls ``load_raw`` then ``build_customer_features``.
        - Prints the raw shape, the engineered shape and the churn rate.
        - Ensures the processed-data directory exists and writes the
          engineered DataFrame to ``OUT_FILE`` as CSV.

    Output:
        ``None``. Side effect: writes ``data/processed/customers.csv``.
    """
    df = load_raw()
    print(f'Raw data loaded: {df.shape}')

    customers = build_customer_features(df)
    print(f'Customer-level features built: {customers.shape}')
    print(f'Churn rate: {customers["churn"].mean():.2%}')

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(OUT_FILE)
    print(f'Saved {customers.shape[0]} customers to {OUT_FILE}')


if __name__ == '__main__':
    main()
