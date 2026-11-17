"""Helper (label-free): single-feature adversarial AUC (fit rows vs validation rows) of the columns that the
current train.make_features returns, on fold 1 and on fold 4. Labels are not used.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402
from train import make_features  # noqa: E402

df = load_train()
X, _ = split_xy(df)
folds = time_ordered_folds(df[DATE_COLUMN])
res = {}
for k in (0, 3):
    fit_idx, val_idx = folds[k]
    A, B = make_features(X.iloc[fit_idx].copy(), X.iloc[val_idx].copy())
    is_late = np.r_[np.zeros(len(A)), np.ones(len(B))]
    for col in A.columns:
        v = np.nan_to_num(np.r_[A[col].values, B[col].values].astype(float), nan=-1e9)
        res.setdefault(col, []).append(roc_auc_score(is_late, v))
for col, (a, b) in sorted(res.items(), key=lambda kv: -max(abs(x - 0.5) for x in kv[1]))[:30]:
    print(f'{col:45s} fold1={a:.3f} fold4={b:.3f}')
