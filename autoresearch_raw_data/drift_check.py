"""Helper: label-free adversarial check. How well does each feature of train.make_features
separate the first 60% of customers (by time) from the last 40%? (Single-feature AUC, |AUC-0.5|.)
Labels are not used.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import customer_table, load_lines, split_lines, time_ordered_folds  # noqa: E402
from train import make_features  # noqa: E402

lines, y = split_lines(load_lines())
customers = customer_table(load_lines())
fit_pos, val_pos = time_ordered_folds(customers['first_date'])[0]
fit_ids, val_ids = y.index[fit_pos], y.index[val_pos]
X_fit, X_val = make_features(lines[lines['customer_id'].isin(fit_ids)].copy(), y.loc[fit_ids].copy(),
                             lines[lines['customer_id'].isin(val_ids)].copy())
is_late = np.r_[np.zeros(len(X_fit)), np.ones(len(X_val))]
for col in X_fit.columns:
    values = np.r_[X_fit[col].fillna(-1).values, X_val[col].fillna(-1).values]
    auc = roc_auc_score(is_late, values)
    print(f'{col:28s} adv_auc={auc:.3f}  early_median={X_fit[col].median():.3f}  late_median={X_val[col].median():.3f}')
