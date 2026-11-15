"""HUMAN-ONLY final evaluation. The agent must never read, run or reference this.

Takes the ``make_features`` / ``make_model`` currently in ``train.py``, fits on all
training customers, evaluates ONCE on the held-out test customers, and writes
``results/autoresearch_raw_data.json`` in the same shape as
``results/conventional_xgboost.json``.

    python autoresearch_raw_data/final_eval.py --tag <run_tag> --n-experiments 100 --wall-clock-hours <H>
"""
import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import (PROJECT_ROOT, RANDOM_SEED, TRAIN_FRACTIONS, build_features,  # noqa: E402
                     load_lines, run_cv, split_lines)
from train import make_features, make_model  # noqa: E402

TEST_FILE = PROJECT_ROOT / 'data' / 'processed' / 'raw_test.csv'
RESULTS_FILE = PROJECT_ROOT / 'results' / 'autoresearch_raw_data.json'
THRESHOLD = 0.5


def main(tag: str, n_experiments, wall_clock_hours) -> None:
    cv = run_cv(make_features, make_model)

    lines_train, y_train = split_lines(load_lines())
    lines_test, y_test = split_lines(load_lines(TEST_FILE))
    X_fit, X_eval = build_features(make_features, lines_train, y_train, lines_test, y_test.index)
    model = make_model()
    model.fit(X_fit, y_train)

    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    output = {
        'approach': 'autoresearch_raw_data',
        'run_tag': tag,
        'random_seed': RANDOM_SEED,
        'n_experiments': n_experiments,
        'agent_wall_clock_hours': wall_clock_hours,
        'cv': {
            'type': 'time-ordered expanding window',
            'train_fractions': list(TRAIN_FRACTIONS),
            'objective': 'mean validation AUC-ROC',
            'best_mean_auc': cv['cv_auc'],
            'best_fold_aucs': cv['fold_aucs'],
        },
        'test_metrics': {
            'threshold': THRESHOLD,
            'accuracy': float(accuracy_score(y_test, pred)),
            'precision': float(precision_score(y_test, pred, zero_division=0)),
            'recall': float(recall_score(y_test, pred, zero_division=0)),
            'f1': float(f1_score(y_test, pred, zero_division=0)),
            'auc_roc': float(roc_auc_score(y_test, proba)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        },
        'train_size': len(X_fit),
        'test_size': len(X_eval),
        'n_features': X_fit.shape[1],
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['test_metrics'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n-experiments', type=int, default=None)
    ap.add_argument('--wall-clock-hours', type=float, default=None)
    a = ap.parse_args()
    main(a.tag, a.n_experiments, a.wall_clock_hours)
