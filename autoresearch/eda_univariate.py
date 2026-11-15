"""EDA helper (not an experiment): univariate AUC of each feature on the EARLIEST 60%
of training customers only (= fold-1 training window), so no validation-window
labels influence feature ideas.

Usage: python3 autoresearch/eda_univariate.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prepare import load_train, split_xy, time_ordered_folds

df = load_train()
X, y = split_xy(df)
fit_idx, _ = time_ordered_folds(df['first_date'])[0]
X, y = X.iloc[fit_idx], y.iloc[fit_idx]
print(f'rows used: {len(X)}, churn rate {y.mean():.3f}')

rows = []
for col in X.columns:
    x = X[col]
    nan_rate = x.isna().mean()
    filled = x.fillna(x.median())
    if filled.nunique() < 2:
        continue
    auc = roc_auc_score(y, filled)
    rows.append((col, max(auc, 1 - auc), '+' if auc >= 0.5 else '-', nan_rate))
out = pd.DataFrame(rows, columns=['feature', 'auc', 'dir', 'nan_rate'])
pd.set_option('display.width', 200)
print(out.sort_values('auc', ascending=False).head(40).to_string(index=False))
print('\nNaN rate summary:', out.loc[out.nan_rate > 0, ['feature', 'nan_rate']].head(10).to_string(index=False))
print('\nall-NaN-prior rows share:', X.filter(like='prior_').isna().all(axis=1).mean().round(3))
