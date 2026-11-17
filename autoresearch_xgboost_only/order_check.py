"""Helper: are the rows of train.csv in time order? (label-free; uses first_date only for ordering)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import DATE_COLUMN, load_train  # noqa: E402

d = load_train()[DATE_COLUMN]
print('monotonic increasing:', d.is_monotonic_increasing)
print('fraction of adjacent pairs in order:', (d.diff().dropna() >= 0).mean())
print('range', d.min(), d.max())
