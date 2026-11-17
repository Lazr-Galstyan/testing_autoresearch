"""Helper: quick EDA of raw_train.csv (the only data the agent may read)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import load_lines  # noqa: E402

df = load_lines()
print(df.dtypes)
print(df.describe(include='all').T)
c = df.groupby('customer_id').agg(first=('invoice_date', 'min'), churn=('churn', 'first'),
                                  country=('country', 'first'), n_inv=('invoice', 'nunique'))
print('churn rate', c.churn.mean(), 'date range', c['first'].min(), c['first'].max())
print(c.groupby(c['first'].dt.to_period('M')).churn.agg(['mean', 'size']))
print(c.groupby('country').churn.agg(['mean', 'size']).sort_values('size', ascending=False).head(12))
print(c.groupby(c['first'].dt.hour).churn.agg(['mean', 'size']))
print(c.groupby(c['first'].dt.dayofweek).churn.agg(['mean', 'size']))
print('n_inv dist', c.n_inv.value_counts().head())
print('neg qty lines', (df.quantity < 0).sum(), 'zero price', (df.price <= 0).sum())
print(df.invoice.astype(str).str[0].value_counts().head())
print(df.stock_code.astype(str).str.match(r'^\d').value_counts())
print(df[~df.stock_code.astype(str).str.match(r'^\d')].stock_code.value_counts().head(15))
