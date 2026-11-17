"""Helper: run one experiment, print metric + feature list, then keep or revert.

keep  -> CV AUC strictly higher than the best `keep` row in results.tsv: copy train.py to best_train.py
else  -> copy best_train.py back over train.py (discard or crash)
Logging to results.tsv / research-log.md is done by hand afterwards.
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN, BEST, LOG = HERE / 'train.py', HERE / 'best_train.py', HERE / 'run.log'

with open(HERE / 'results.tsv') as fh:
    rows = list(csv.DictReader(fh, delimiter='\t'))
keeps = [float(r['CV_AUC']) for r in rows if r['status'] == 'keep']
best = max(keeps) if keeps else float('-inf')
print(f'experiment #{len(rows) + 1}, best so far {best:.4f}')

feat = subprocess.run([sys.executable, str(HERE / 'show_features.py')], cwd=ROOT,
                      capture_output=True, text=True, timeout=120)
print(feat.stdout.strip() or feat.stderr[-1500:])

try:
    out = subprocess.run([sys.executable, str(TRAIN)], cwd=ROOT, capture_output=True,
                         text=True, timeout=120)
    text = out.stdout + out.stderr
except subprocess.TimeoutExpired:
    text = 'TIMEOUT'
LOG.write_text(text)
print('\n'.join(l for l in text.splitlines() if l.startswith(('Fold', 'CV'))))
auc_line = [l for l in text.splitlines() if l.startswith('CV AUC:')]
if not auc_line:
    print('CRASH\n' + text[-2000:])
    if BEST.exists():
        shutil.copy(BEST, TRAIN)
    sys.exit(0)
auc = float(auc_line[0].split(':')[1])
if auc > best:
    shutil.copy(TRAIN, BEST)
    print(f'KEEP ({auc:.4f} > {best:.4f})')
else:
    shutil.copy(BEST, TRAIN)
    print(f'DISCARD ({auc:.4f} <= {best:.4f}); train.py restored')
