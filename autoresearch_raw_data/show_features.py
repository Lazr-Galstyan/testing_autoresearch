"""Helper: print the feature columns produced by train.make_features (comma-separated).

Uses a time-ordered split of raw_train.csv only (first 60% of customers as fit),
the same way the harness does, so nothing extra is read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare import customer_table, load_lines, split_lines, time_ordered_folds  # noqa: E402
from train import make_features  # noqa: E402

lines, y = split_lines(load_lines())
customers = customer_table(load_lines())
fit_pos, val_pos = time_ordered_folds(customers['first_date'])[0]
fit_ids, val_ids = y.index[fit_pos], y.index[val_pos]
X_fit, _ = make_features(lines[lines['customer_id'].isin(fit_ids)].copy(), y.loc[fit_ids].copy(),
                         lines[lines['customer_id'].isin(val_ids)].copy())
print(f'n_features={X_fit.shape[1]}')
print(', '.join(X_fit.columns))
