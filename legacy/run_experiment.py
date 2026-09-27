#!/usr/bin/env python3
"""Unified experiment runner: blocking stats + train + validation in a single script.

Usage:
  python3 run_experiment.py --sample-s1 10000 --top-k 40 --max-df 5000 --exp-id EXP-001
  python3 run_experiment.py --sample-s1 50000 --top-k 60 --max-df 10000 --exp-id EXP-002

This avoids re-loading the large pickle files multiple times.
"""
import sys
sys.path.insert(0, "/Users/himanshijain/amazon_ml_er_solution/code/business_entity_resolution/src")

import argparse
import json
import time
import resource
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ber.blocking import Blocker, blocking_stats
from ber.features import FEATURE_NAMES, FeatureContext, pair_features
from ber.io import read_ground_truth
from ber.metric import fast_macro_f05, oracle_ceiling, per_entity_f05
from ber.normalize import normalize_frame
from ber import decide as D
from ber import model as M

WORK = Path("/Users/himanshijain/amazon_ml_er_solution/work")


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} | RSS {rss_gb():.2f} GB] {msg}", flush=True)


def _bucket(s1_id: str) -> int:
    return int(hashlib.md5(s1_id.encode()).hexdigest()[:8], 16) % 100


def load_split(split):
    s1 = pd.read_pickle(WORK / f"{split}_s1.pkl")
    pool = pd.concat([pd.read_pickle(WORK / f"{split}_s2.pkl"),
                      pd.read_pickle(WORK / f"{split}_s3.pkl")], ignore_index=True)
    return s1, pool


def select_s1(s1, gt, sample, seed=42):
    if sample and sample < len(s1):
        s1 = s1.sample(n=sample, random_state=seed)
    s1 = s1.reset_index(drop=True)
    gtl = [gt.get(x, []) for x in s1["entity_id"].values]
    return s1, gtl


def generate(blocker, ctx, s1, pool, with_features=True, chunk=5000):
    I, J, S, X = [], [], [], []
    for i, j, sn, sa in blocker.query(s1, chunk=chunk):
        s = np.maximum(sn, sa)
        o = np.lexsort((-s, i))
        i, j, sn, sa, s = i[o], j[o], sn[o], sa[o], s[o]
        I.append(i); J.append(j); S.append(s)
        if with_features:
            X.append(pair_features(ctx, i, j, sn, sa))
    if not I:
        e = np.zeros(0, np.int32)
        return e, e, np.zeros(0, np.float32), np.zeros((0, len(FEATURE_NAMES)), np.float32)
    return (np.concatenate(I), np.concatenate(J), np.concatenate(S),
            np.vstack(X) if with_features else None)


def labels_for(i, j, pool_ids, gtl):
    own = {}
    for k, g in enumerate(gtl):
        for x in g:
            own.setdefault(x, set()).add(k)
    cid = pool_ids[j]
    return np.fromiter((k in own.get(c, ()) for k, c in zip(i.tolist(), cid)), bool, len(i))


def run(args):
    t0 = time.time()
    log("Loading data...")
    s1_all, pool = load_split("train")
    gt = pd.read_pickle(WORK / "train_gt.pkl")
    s1, gtl = select_s1(s1_all, gt, args.sample_s1)
    del s1_all
    n_true = np.array([len(g) for g in gtl], float)
    log(f"S1={len(s1):,} pool={len(pool):,} avg_matches={n_true[n_true>0].mean():.2f} singletons={int((n_true==0).sum()):,}")
    
    # Split into train/val/es
    b = np.array([_bucket(x) for x in s1["entity_id"].values])
    role = np.where(b < 20, "val", np.where(b < 30, "es", "fit"))
    log(f"Roles: {pd.Series(role).value_counts().to_dict()}")
    
    # Build blocker
    log(f"Building blocker: top_k={args.top_k}, max_df={args.max_df}, min_score={args.min_score}")
    blk = Blocker(n_features=2**args.hash_bits, max_df=args.max_df, top_k=args.top_k, 
                  min_score=args.min_score, nnz_budget=args.nnz_budget)
    blk.fit_pool(pool)
    
    # Build features
    ctx = FeatureContext(s1, pool)
    log("Feature context ready, generating candidates...")
    i, j, s, X = generate(blk, ctx, s1, pool)
    pool_ids = pool["entity_id"].values
    y = labels_for(i, j, pool_ids, gtl)
    log(f"Pairs={len(i):,} pos={y.sum():,} ({y.mean()*100:.3f}%) neg/pos={(~y).sum()/max(y.sum(),1):.1f}")
    
    # Blocking stats
    st = blocking_stats(i, pool_ids[j].tolist(), len(s1), gtl, n_pool=len(pool))
    st["oracle_macro_f05_ceiling"] = oracle_ceiling(i, y, n_true, len(s1))
    log(f"BLOCKING: recall={st.get('pair_recall',0):.4f} entity_recall={st.get('macro_entity_recall',0):.4f} "
        f"full_entity_recall={st.get('entities_full_recall',0):.4f} oracle_f05={st['oracle_macro_f05_ceiling']:.4f}")
    log(f"  mean_cands={st['mean_cands']:.1f} p95={st['p95_cands']:.0f} max={st['max_cands']} "
        f"zero_cand={st['zero_cand_s1_frac']:.4f}")
    
    # Train model
    r = role[i]
    log("Training LightGBM...")
    bst = M.train(X[r == "fit"], y[r == "fit"], X[r == "es"], y[r == "es"], 
                  scale_pos_weight=args.scale_pos_weight)
    log(f"LightGBM best_iteration={bst.best_iteration}")
    
    # Validation
    vmask = r == "val"
    val_rows = np.flatnonzero(role == "val")
    remap = np.full(len(s1), -1)
    remap[val_rows] = np.arange(len(val_rows))
    g = remap[i[vmask]]
    val_n_true = np.array([len(gtl[k]) for k in val_rows], float)
    prob = M.predict(bst, X[vmask])
    yv, cand, is_s3 = y[vmask], j[vmask], X[vmask][:, FEATURE_NAMES.index("is_s3")]
    
    info = json.loads((WORK / "gt_structure.json").read_text())
    best = D.tune(g, cand, prob, is_s3, yv, val_n_true, len(val_rows), allow_o2o=info["one_to_one"])
    keep = D.decide(g, cand, prob, is_s3, len(val_rows), best["t2"], best["t3"], best["rel"], best["o2o"])
    pe = per_entity_f05(g, keep, yv, val_n_true, len(val_rows))
    ceiling = oracle_ceiling(g, yv, val_n_true, len(val_rows))
    
    vc = s1["country"].values[val_rows]
    by_country = {c: float(pe[vc == c].mean()) for c in np.unique(vc)}
    single = val_n_true == 0
    npred = np.bincount(g[keep], minlength=len(val_rows))
    
    pair_prec = float((keep & yv).sum() / max(keep.sum(), 1))
    pair_rec = float((keep & yv).sum() / max(val_n_true.sum(), 1))
    singleton_acc = float((npred[single] == 0).mean()) if single.any() else None
    multi_f05 = float(pe[val_n_true >= 2].mean())
    
    result = {
        "exp": args.exp_id,
        "sample_s1": args.sample_s1,
        "top_k": args.top_k,
        "max_df": args.max_df,
        "min_score": args.min_score,
        "scale_pos_weight": args.scale_pos_weight,
        "val_s1": int(len(val_rows)),
        "val_macro_f05": float(pe.mean()),
        "blocking_ceiling_f05": ceiling,
        "blocking_pair_recall": st.get("pair_recall", 0),
        "blocking_entity_recall": st.get("macro_entity_recall", 0),
        "decision": best,
        "by_country": by_country,
        "singleton_accuracy": singleton_acc,
        "multi_match_f05": multi_f05,
        "pair_precision": pair_prec,
        "pair_recall": pair_rec,
        "n_pairs": int(len(i)),
        "mean_cands": st["mean_cands"],
        "p95_cands": st["p95_cands"],
        "max_cands": st["max_cands"],
        "pos_rate": float(y.mean()),
        "lgbm_iterations": bst.best_iteration,
        "runtime_s": time.time() - t0,
        "peak_rss_gb": rss_gb(),
    }
    
    log("=" * 70)
    log(f"RESULT {args.exp_id}")
    log(f"  Macro F0.5:          {result['val_macro_f05']:.4f}")
    log(f"  Blocking ceiling:    {result['blocking_ceiling_f05']:.4f}")
    log(f"  Pair precision:      {result['pair_precision']:.4f}")
    log(f"  Pair recall:         {result['pair_recall']:.4f}")
    log(f"  Singleton accuracy:  {result['singleton_accuracy']}")
    log(f"  Multi-match F0.5:    {result['multi_match_f05']:.4f}")
    log(f"  By country:          {result['by_country']}")
    log(f"  Decision:            {result['decision']}")
    log(f"  Blocking recall:     {result['blocking_pair_recall']:.4f}")
    log(f"  Mean cands:          {result['mean_cands']:.1f}")
    log(f"  P95 cands:           {result['p95_cands']:.0f}")
    log(f"  Runtime:             {result['runtime_s']:.0f}s")
    log(f"  Peak RSS:            {result['peak_rss_gb']:.2f} GB")
    log("=" * 70)
    
    # Save
    with open(WORK / "experiments.jsonl", "a") as f:
        f.write(json.dumps(result, default=float) + "\n")
    
    M.save(bst, WORK / "model.txt", {"decision": best, 
           "blocker": dict(top_k=args.top_k, max_df=args.max_df,
                          min_score=args.min_score, hash_bits=args.hash_bits),
           "validation": result})
    
    # Dump errors for analysis
    if args.dump_errors:
        _dump_errors(s1, pool, val_rows, g, cand, prob, keep, yv, gtl, X[vmask], i, j, pool_ids)
    
    return result


def _dump_errors(s1, pool, val_rows, g, cand, prob, keep, yv, gtl, Xv, all_i, all_j, pool_ids):
    """Dump detailed error analysis."""
    fi = {n: k for k, n in enumerate(FEATURE_NAMES)}
    rows = []
    for m in np.flatnonzero(keep != yv):
        k = val_rows[g[m]]
        n_true_k = len(gtl[k])
        kind = ("FP_singleton" if n_true_k == 0 else "FP") if keep[m] else "FN_scored"
        rows.append(dict(
            kind=kind, s1=s1.entity_id.values[k], cand=pool.entity_id.values[cand[m]],
            prob=float(prob[m]), n_true=n_true_k,
            s1_name=s1.business_name.values[k], cand_name=pool.business_name.values[cand[m]],
            s1_addr=s1.business_address.values[k], cand_addr=pool.business_address.values[cand[m]],
            core_ratio=float(Xv[m, fi["core_ratio"]]),
            addr_ratio=float(Xv[m, fi["addr_ratio"]]),
            core_jw=float(Xv[m, fi["core_jw"]]),
            name_idf_cos=float(Xv[m, fi["name_idf_cos"]]),
        ))
    
    # Also find blocking misses
    cand_set = set(zip(g.tolist(), pool.entity_id.values[cand].tolist()))
    pid = pd.Series(np.arange(len(pool)), index=pool.entity_id.values)
    for gi, k in enumerate(val_rows):
        for x in gtl[k]:
            if (gi, x) not in cand_set:
                c = pid[x]
                rows.append(dict(
                    kind="FN_blocking_miss", s1=s1.entity_id.values[k], cand=x, prob=None,
                    n_true=len(gtl[k]), s1_name=s1.business_name.values[k],
                    cand_name=pool.business_name.values[c], s1_addr=s1.business_address.values[k],
                    cand_addr=pool.business_address.values[c],
                ))
    df = pd.DataFrame(rows)
    df.to_csv(WORK / "val_errors.tsv", sep="\t", index=False)
    
    # Error summary
    if len(df) > 0:
        log(f"Errors: {len(df):,} total")
        log(f"  {df['kind'].value_counts().to_dict()}")
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-s1", type=int, default=10000)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-df", type=int, default=5000)
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--hash-bits", type=int, default=20)
    ap.add_argument("--scale-pos-weight", type=float, default=1.0)
    ap.add_argument("--nnz-budget", type=int, default=20_000_000)
    ap.add_argument("--exp-id", default="EXP-001")
    ap.add_argument("--dump-errors", action="store_true")
    args = ap.parse_args()
    run(args)
