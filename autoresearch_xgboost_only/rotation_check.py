"""Helper: check whether the rotated feature matrices of train.RotatedPCAEnsemble contain inf/NaN
(the matmul RuntimeWarnings may be spurious BLAS warnings). Uses fold 1 fit rows only; labels are needed
only because fit() trains the members.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402
from train import make_features, make_model  # noqa: E402

df = load_train()
X, y = split_xy(df)
fit_idx, val_idx = time_ordered_folds(df[DATE_COLUMN])[0]
X_fit, X_val = make_features(X.iloc[fit_idx].copy(), X.iloc[val_idx].copy())
model = make_model().fit(X_fit, y.iloc[fit_idx])
for name, data in (('fit', X_fit), ('val', X_val)):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        R = model._rotate(data, model.members[0][0]).to_numpy(dtype=float)
    finite = np.isfinite(R[~np.isnan(R)]).all()
    print(f'{name}: shape={R.shape} nan={np.isnan(R).sum()} all_non_nan_finite={finite} '
          f'max_abs={np.nanmax(np.abs(R)):.3g} warnings={len(w)}')
    # recompute the same products in float64 with einsum (no BLAS) to compare
    q, rot = model.members[0][0]
    p = data[model.pca_cols].to_numpy(dtype=float)
    diff = np.abs(np.einsum('ij,jk->ik', p, q) - R[:, [data.columns.get_loc(c) for c in model.pca_cols]]).max()
    print(f'{name}: max |BLAS matmul - einsum| on pca block = {diff:.3g}')
