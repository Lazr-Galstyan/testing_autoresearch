"""Helper for the autoresearch loop (does not touch results.tsv or research-log.md).

Runs train.py, compares the CV AUC with the best `keep` row in results.tsv, then
copies train.py -> best_train.py (keep) or best_train.py -> train.py (discard/crash).
Prints the status line so the agent can log it by hand.

Usage: python3 autoresearch/run_experiment.py
"""
import csv
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    proc = subprocess.run(['python3', 'train.py'], cwd=HERE, capture_output=True,
                          text=True, timeout=120)
    log = proc.stdout + proc.stderr
except subprocess.TimeoutExpired:
    log = 'TIMEOUT (>120s)'
(HERE / 'run.log').write_text(log)

with open(HERE / 'results.tsv') as fh:
    rows = list(csv.DictReader(fh, delimiter='\t'))
# Current best = the most recent `keep` (best_train.py). Usually also the max, but a
# simplification keep (equal AUC, less code; see program.md) can be marginally lower.
best = ([float(r['CV_AUC']) for r in rows if r['status'] == 'keep'] or [0.0])[-1]

auc = re.search(r'^CV AUC: ([\d.]+)', log, re.M)
secs = re.search(r'^CV time: ([\d.]+)s', log, re.M)
folds = re.search(r'^Fold AUCs: (.*)$', log, re.M)
if auc:
    value = float(auc.group(1))
    status = 'keep' if value > best else 'discard'
else:
    value, status = 0.0, 'crash'

if status == 'keep':
    shutil.copy(HERE / 'train.py', HERE / 'best_train.py')
else:
    shutil.copy(HERE / 'best_train.py', HERE / 'train.py')

print(f"CV_AUC={value:.4f} cv_seconds={float(secs.group(1)) if secs else 0.0:.1f} "
      f"status={status} best_before={best:.4f} folds={folds.group(1) if folds else 'n/a'}")
if status == 'crash':
    print(log[-2000:])
