"""Turn pair probabilities into per-S1 match sets. Supports zero / one / many matches.

Decision rule (all parts optional, all tuned on validation macro F0.5):
  keep(pair) = p >= t_src                        (t_src: global, or separate S2 / S3 thresholds)
           and p >= rel * max_p_within_its_S1    (relative-to-best filter)
  o2o: each S2/S3 record is assigned to at most ONE S1 (the S1 where it scores highest).
       Enable only if the training ground truth confirms that no S2/S3 id belongs to >1 S1
       (`gt_is_one_to_one`).
"""
from __future__ import annotations

import itertools
from typing import Dict

import numpy as np

from .metric import fast_macro_f05


def group_max(g, v, n):
    m = np.full(n, -np.inf, np.float32)
    np.maximum.at(m, g, v)
    return m


def decide(g, cand, prob, is_s3, n_groups, t2, t3, rel=0.0, o2o=False):
    th = np.where(is_s3 > 0.5, t3, t2)
    keep = prob >= th
    if rel > 0:
        keep &= prob >= rel * group_max(g, prob, n_groups)[g]
    if o2o and keep.any():
        idx = np.flatnonzero(keep)
        order = idx[np.lexsort((g[idx], -prob[idx], cand[idx]))]
        first = np.r_[True, cand[order][1:] != cand[order][:-1]]
        keep = np.zeros_like(keep)
        keep[order[first]] = True
    return keep


def tune(g, cand, prob, is_s3, label, n_true, n_groups, allow_o2o, min_gain=5e-4, n_eval=None) -> Dict:
    """Grid-search the decision rule on validation macro F0.5.
    Groups 0..n_eval-1 are evaluated; groups >= n_eval are COMPETITOR S1 (competitor closure):
    they take part in the decision (one-to-one) but are never scored."""
    n_eval = n_groups if n_eval is None else n_eval
    ev = g < n_eval
    ts = np.round(np.arange(0.02, 0.99, 0.02), 3)
    rels = [0.0, 0.3, 0.5, 0.7, 0.85]
    o2os = [False, True] if allow_o2o else [False]
    results = []

    def score(t2, t3, r, o):
        k = decide(g, cand, prob, is_s3, n_groups, t2, t3, r, o)
        return fast_macro_f05(g[ev], k[ev], label[ev], n_true, n_eval)

    best = {"f05": -1}
    for t, r, o in itertools.product(ts, rels, o2os):
        f = score(t, t, r, o)
        results.append((float(t), r, o, f))
        if f > best["f05"]:
            best = {"t2": float(t), "t3": float(t), "rel": r, "o2o": o, "f05": f}
    base = dict(best)
    for d2, d3 in itertools.product(np.arange(-0.1, 0.101, 0.02), repeat=2):
        a, b = base["t2"] + d2, base["t3"] + d3
        if not (0.01 < a < 0.99 and 0.01 < b < 0.99):
            continue
        f = score(a, b, base["rel"], base["o2o"])
        if f > max(best["f05"], base["f05"] + min_gain):
            best = {"t2": float(a), "t3": float(b), "rel": base["rel"], "o2o": base["o2o"], "f05": f}
    best["global_rule_f05"] = base["f05"]
    for o in o2os:  # best f05 with o2o forced on / off (for the o2o ablation report)
        best[f"best_f05_o2o_{str(o).lower()}"] = max(x[3] for x in results if x[2] == o)
    return best


def gt_is_one_to_one(gt) -> Dict:
    seen, dup = set(), 0
    for v in gt.values():
        for x in v:
            if x in seen:
                dup += 1
            seen.add(x)
    return {"n_ids": len(seen), "ids_in_multiple_s1": dup, "one_to_one": dup == 0}
