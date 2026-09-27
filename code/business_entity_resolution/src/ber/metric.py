"""Official challenge metric: per-Source-1-entity F_0.5, macro-averaged over ALL S1 entities
(singletons included: empty prediction -> 1.0, any prediction -> 0.0).

`fast_macro_f05` is an exact vectorised equivalent used inside threshold tuning. It takes
`n_true` from the ground truth — NOT from the candidate set — so true matches lost by
blocking count as false negatives, and S1 entities with zero candidates are still scored.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def f05_entity(pred: Iterable[str], true: Iterable[str]) -> float:
    p, t = set(pred), set(true)
    if not t:
        return 1.0 if not p else 0.0
    tp = len(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(t)
    return 1.25 * prec * rec / (0.25 * prec + rec)


def macro_f05(preds: Dict[str, Iterable[str]], gt: Dict[str, Iterable[str]]) -> float:
    if not gt:
        return 0.0
    return sum(f05_entity(preds.get(k, ()), v) for k, v in gt.items()) / len(gt)


def per_entity_f05(g, keep, label, n_true, n_groups):
    keep = keep.astype(bool)
    tp = np.bincount(g[keep & label], minlength=n_groups).astype(np.float64)
    npred = np.bincount(g[keep], minlength=n_groups).astype(np.float64)
    f = np.zeros(n_groups)
    single = n_true == 0
    f[single] = (npred[single] == 0).astype(float)
    ok = (~single) & (tp > 0)
    P = tp[ok] / npred[ok]
    R = tp[ok] / n_true[ok]
    f[ok] = 1.25 * P * R / (0.25 * P + R)
    return f


def fast_macro_f05(g, keep, label, n_true, n_groups) -> float:
    return float(per_entity_f05(g, keep, label, n_true, n_groups).mean()) if n_groups else 0.0


def oracle_ceiling(g, label, n_true, n_groups) -> float:
    """Best achievable macro F0.5 on this candidate set (perfect classifier)."""
    return fast_macro_f05(g, label.astype(bool), label.astype(bool), n_true, n_groups)


def full_report(g, keep, label, n_true, n_groups):
    """All decision-quality metrics on evaluated groups (official metric first)."""
    keep, label = keep.astype(bool), label.astype(bool)
    f = per_entity_f05(g, keep, label, n_true, n_groups)
    tp = np.bincount(g[keep & label], minlength=n_groups).astype(float)
    npred = np.bincount(g[keep], minlength=n_groups).astype(float)
    has_pred, has_true = npred > 0, n_true > 0
    single = ~has_true
    return {
        "macro_f05": float(f.mean()),
        "macro_precision": float((tp[has_pred] / npred[has_pred]).mean()) if has_pred.any() else 1.0,
        "macro_recall": float((tp[has_true] / n_true[has_true]).mean()) if has_true.any() else 1.0,
        "micro_precision": float(tp.sum() / max(npred.sum(), 1)),
        "micro_recall": float(tp.sum() / max(n_true.sum(), 1)),
        "fp_pairs": int(npred.sum() - tp.sum()),
        "fp_rate_of_predictions": float(1 - tp.sum() / max(npred.sum(), 1)),
        "singleton_accuracy": float((npred[single] == 0).mean()) if single.any() else None,
        "singleton_f05": float(f[single].mean()) if single.any() else None,
        "multi_match_f05": float(f[n_true >= 2].mean()) if (n_true >= 2).any() else None,
        "one_match_f05": float(f[n_true == 1].mean()) if (n_true == 1).any() else None,
        "mean_pred_per_s1": float(npred.mean()),
        "s1_with_no_pred": float((npred == 0).mean()),
    }, f
