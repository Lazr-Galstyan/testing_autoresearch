"""Helper: print the feature columns produced by train.make_features (comma-separated).

Uses the first time-ordered fold of data/processed/train.csv (via the harness), so nothing
extra is read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402
from train import make_features  # noqa: E402

df = load_train()
X, y = split_xy(df)
fit_idx, val_idx = time_ordered_folds(df[DATE_COLUMN])[0]
X_fit, _ = make_features(X.iloc[fit_idx].copy(), X.iloc[val_idx].copy())
print(f'n_features={X_fit.shape[1]}')
print(', '.join(X_fit.columns))
