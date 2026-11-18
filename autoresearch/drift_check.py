"""Helper (label-free): adversarial drift check per feature group. For fold 1, how well does each single feature
separate fit rows (early customers) from validation rows (later customers)? AUC far from 0.5 = drifting.
Labels are not used.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402

df = load_train()
X, _ = split_xy(df)
fit_idx, val_idx = time_ordered_folds(df[DATE_COLUMN])[0]
is_late = np.r_[np.zeros(len(fit_idx)), np.ones(len(val_idx))]
rows = []
for col in X.columns:
    v = np.r_[X[col].iloc[fit_idx].values, X[col].iloc[val_idx].values].astype(float)
    v = np.nan_to_num(v, nan=-1e9)
    a = roc_auc_score(is_late, v)
    rows.append((abs(a - 0.5), a, col))
for _, a, col in sorted(rows, reverse=True)[:45]:
    print(f'{col:50s} adv_auc={a:.3f}')
