"""Helper (label-free): do cumulative prior_* counts lose their drift when divided by days elapsed since the
start of the fit data? Elapsed days come from year/month/day_of_month (first_date is not used).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402

df = load_train()
X, _ = split_xy(df)
fit_idx, val_idx = time_ordered_folds(df[DATE_COLUMN])[0]
day = pd.to_datetime(dict(year=X['year'], month=X['month'], day=X['day_of_month']))
elapsed = (day - day.iloc[fit_idx].min()).dt.days + 30
is_late = np.r_[np.zeros(len(fit_idx)), np.ones(len(val_idx))]
for col in [c for c in X.columns if c.startswith(('prior_transactions_', 'prior_units_'))]:
    for name, v in ((col, X[col]), (col + ' / elapsed', X[col] / elapsed)):
        a = np.r_[v.iloc[fit_idx].values, v.iloc[val_idx].values]
        print(f'{name:45s} adv_auc={roc_auc_score(is_late, a):.3f}')
