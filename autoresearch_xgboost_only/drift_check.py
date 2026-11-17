"""Helper: label-free adversarial check on the raw train.csv features.

For each feature: single-feature AUC for separating the first 60% of customers (by time)
from the last 40% (|AUC - 0.5| = drift). Labels are not used. Also prints basic stats.
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
    v = np.r_[X[col].iloc[fit_idx].fillna(-999).values, X[col].iloc[val_idx].fillna(-999).values]
    auc = roc_auc_score(is_late, v) if np.unique(v).size > 1 else 0.5
    rows.append((abs(auc - 0.5), col, auc, X[col].isna().mean(), X[col].nunique()))
for d, col, auc, na, nu in sorted(rows, reverse=True):
    print(f'{col:45s} adv_auc={auc:.3f} na={na:.2f} nunique={nu}')
