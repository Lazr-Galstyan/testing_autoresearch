"""Helper: fit the current train.py pipeline on the fit rows of the last fold and print the largest standardised
LR coefficients (fit rows only; validation rows are not touched).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402
from train import make_features, make_model  # noqa: E402

df = load_train()
X, y = split_xy(df)
fit_idx, val_idx = time_ordered_folds(df[DATE_COLUMN])[-1]
X_fit, _ = make_features(X.iloc[fit_idx].copy(), X.iloc[val_idx].copy())
model = make_model().fit(X_fit, y.iloc[fit_idx])
lr = model[-1]
names = list(X_fit.columns) + [f'{X_fit.columns[i]}_missing' for i in model[1].indicator_.features_]
coef = lr.coef_[0]
for i in np.argsort(-np.abs(coef))[:40]:
    print(f'{names[i]:50s} {coef[i]:+.4f}')
