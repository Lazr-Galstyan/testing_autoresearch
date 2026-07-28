"""Build the first-transaction churn dataset from the raw Online Retail II data.

What this produces
------------------
One modelling table where, for each customer, we keep **only their first
transaction** (every line item of their earliest invoice) and attach a binary
``churn`` label describing whether they churn within the next 90 days.

Churn definition (90-day window)
--------------------------------
A "purchase" is a unique invoice, ordered by date (cancellation invoices still
count). For each customer we look at the gap between their 1st and 2nd purchase:

- ``churn = 0`` — the customer made a 2nd purchase **within** 90 days of the 1st
  (they stayed).
- ``churn = 1`` — the customer made **no** 2nd purchase, **or** made it **more
  than** 90 days after the 1st.

Construction steps (mirrors the EDA notebook's first-transaction section)
-------------------------------------------------------------------------
1. Load the raw line-item data and normalise the column names.
2. Drop rows with a missing ``customer_id`` (cannot be attributed to a customer).
3. Rank each customer's invoices by date to find the 1st and 2nd purchases.
4. Right-censoring: drop customers whose 1st purchase falls within the last
   90 days of the dataset — a full 90-day window has not elapsed, so their label
   cannot be determined.
5. Assign the ``churn`` label from the 1st -> 2nd purchase gap.
6. Keep every line-item row of each eligible customer's first invoice and attach
   the (per-customer) ``churn`` label.

No other cleaning is applied — cancellations, zero prices and odd stock
codes/descriptions are intentionally left in (see the EDA's data-quality issues).

Run from anywhere:
    python src/prepare_churn_data.py

Writes ``data/processed/first_transaction_churn.csv``.
"""
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
CHURN_WINDOW_DAYS = 90

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / 'data' / 'raw' / 'online_retail_II.csv'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
OUT_FILE = PROCESSED_DIR / 'first_transaction_churn.csv'


def load_raw() -> pd.DataFrame:
    """Load the raw Online Retail II CSV and normalise its columns.

    Reads ``RAW_FILE``, lower-cases / renames the columns to snake_case,
    parses ``invoice_date`` and adds ``line_total = quantity * price``.
    """
    df = pd.read_csv(
        RAW_FILE,
        dtype={'Customer ID': 'Int64'},
        parse_dates=['InvoiceDate'],
        encoding='ISO-8859-1',
    )
    df = df.rename(columns={
        'Invoice': 'invoice',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'Price': 'price',
        'Customer ID': 'customer_id',
        'Country': 'country',
    })
    df['line_total'] = df['quantity'] * df['price']
    return df


def prepare_churn_data() -> pd.DataFrame:
    """Return the first-transaction churn table.

    We take only each customer's **first transaction** (every line item of their
    earliest invoice) and label whether they **churn within the next 90 days**
    (no 2nd purchase within 90 days of the 1st, or a 2nd purchase more than
    90 days later). See the module docstring for the full definition and steps.

    The returned frame has the same columns as the raw line-item data plus a
    per-customer ``churn`` label, restricted to each eligible customer's first
    invoice.
    """
    df = load_raw()

    # 2. Only attributable rows can be tied to a customer's first transaction.
    attributed = df.dropna(subset=['customer_id']).copy()

    # 3. Rank each customer's invoices by date (a "purchase" = a unique invoice).
    invoice_seq = (attributed
                   .groupby(['customer_id', 'invoice'])['invoice_date']
                   .min()
                   .reset_index()
                   .sort_values(['customer_id', 'invoice_date']))
    invoice_seq['purchase_rank'] = invoice_seq.groupby('customer_id').cumcount() + 1

    first = invoice_seq[invoice_seq['purchase_rank'] == 1].set_index('customer_id')
    second = invoice_seq[invoice_seq['purchase_rank'] == 2].set_index('customer_id')

    # 4. Right-censoring: a 1st purchase needs a full 90-day window to be labelable.
    dataset_end = df['invoice_date'].max()
    cutoff = dataset_end - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    eligible = first[first['invoice_date'] <= cutoff].copy()

    eligible['second_date'] = second['invoice_date']  # NaT when there is no 2nd purchase
    eligible['days_to_second'] = (
        (eligible['second_date'] - eligible['invoice_date']).dt.total_seconds() / 86400
    )

    # 5. churn = 0 only if a 2nd purchase happened within the window; else churn = 1.
    stayed = (eligible['days_to_second'] <= CHURN_WINDOW_DAYS).fillna(False)
    eligible['churn'] = (~stayed).astype(int)

    # 6. Keep all line-item rows of each eligible customer's first invoice + churn.
    first_invoices = eligible.reset_index()[['customer_id', 'invoice']]
    first_txn = attributed.merge(first_invoices, on=['customer_id', 'invoice'], how='inner')
    first_txn['churn'] = first_txn['customer_id'].map(eligible['churn'])

    return first_txn


def main() -> None:
    """Build the dataset and write it to ``data/processed/first_transaction_churn.csv``."""
    first_txn = prepare_churn_data()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    first_txn.to_csv(OUT_FILE, index=False)

    n_customers = first_txn['customer_id'].nunique()
    churn_rate = first_txn.drop_duplicates('customer_id')['churn'].mean()
    print(f'First-transaction rows : {len(first_txn):,} (across {n_customers:,} customers)')
    print(f'Churn rate             : {churn_rate:.2%}')
    print(f'Saved -> {OUT_FILE}')


if __name__ == '__main__':
    main()
