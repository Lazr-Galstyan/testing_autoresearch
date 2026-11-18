"""Helper: run train.py once, then print the CV AUC, the time, the COMPLETE feature list the model saw
(columns returned by make_features on fold 1) and the keep/discard decision against the best `keep` row of
results.tsv. Usage: python3 autoresearch/run_exp.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

log = HERE / 'run.log'
with open(log, 'w') as f:
    subprocess.run([sys.executable, str(HERE / 'train.py')], stdout=f, stderr=subprocess.STDOUT, timeout=600)
text = log.read_text()
auc = re.search(r'^CV AUC: ([0-9.]+)', text, re.M)
secs = re.search(r'^CV time: ([0-9.]+)s', text, re.M)
folds = re.search(r'^Fold AUCs: (.*)$', text, re.M)
if not auc:
    print('CRASH'); print('\n'.join(text.splitlines()[-30:]))
    sys.exit(1)

from prepare import DATE_COLUMN, load_train, split_xy, time_ordered_folds  # noqa: E402
from train import make_features  # noqa: E402

df = load_train()
X, _ = split_xy(df)
fit_idx, val_idx = time_ordered_folds(df[DATE_COLUMN])[0]
X_fit, _ = make_features(X.iloc[fit_idx].copy(), X.iloc[val_idx].copy())

best = 0.0
with open(HERE / 'results.tsv') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['status'] == 'keep':
            best = max(best, float(row['CV_AUC']))
cv = float(auc.group(1))
print(f'Fold AUCs: {folds.group(1)}')
print(f'CV AUC: {cv:.4f}  CV seconds: {float(secs.group(1)):.1f}  best so far: {best:.4f}  '
      f'-> {"KEEP" if cv > best else "DISCARD"}')
print(f'n_features: {X_fit.shape[1]}')
print('VARIABLES: ' + ','.join(X_fit.columns))
