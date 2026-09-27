#!/usr/bin/env python3
"""1K real-data blocking benchmark against the FULL S2/S3 pool.
Measures peak RSS, runtime, candidate stats, and blocking recall."""
import sys
sys.path.insert(0, "/Users/himanshijain/amazon_ml_er_solution/code/business_entity_resolution/src")

import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/Users/himanshijain/amazon_ml_er_solution/work")


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} RSS={rss_gb():.2f}GB] {msg}", flush=True)


def main():
    sample_s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    max_df = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    t0 = time.time()
    log("Loading pool (S2+S3)...")
    s2 = pd.read_pickle(WORK / "train_s2.pkl")
    log(f"  S2: {len(s2):,}")
    s3 = pd.read_pickle(WORK / "train_s3.pkl")
    log(f"  S3: {len(s3):,}")
    pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    log(f"  pool: {len(pool):,}")

    log("Loading S1 + ground truth...")
    s1_all = pd.read_pickle(WORK / "train_s1.pkl")
    gt = pd.read_pickle(WORK / "train_gt.pkl")

    s1 = s1_all.sample(n=min(sample_s1, len(s1_all)), random_state=42).reset_index(drop=True)
    del s1_all
    gc.collect()
    gtl = [gt.get(x, []) for x in s1["entity_id"].values]
    n_true = np.array([len(g) for g in gtl], float)
    log(f"S1={len(s1):,} pool={len(pool):,} avg_matches={n_true[n_true>0].mean():.2f}")

    from ber.blocking import Blocker, blocking_stats

    log(f"Building blocker: top_k={top_k}, max_df={max_df}")
    blk = Blocker(n_features=2**20, max_df=max_df, top_k=top_k)
    blk.fit_pool(pool)
    log("Index built.")

    log("Querying...")
    tq = time.time()
    I, J, SN, SA = [], [], [], []
    for i, j, sn, sa in blk.query(s1, chunk=1000):
        I.append(i); J.append(j); SN.append(sn); SA.append(sa)

    if I:
        i = np.concatenate(I); j = np.concatenate(J)
        sn = np.concatenate(SN); sa = np.concatenate(SA)
    else:
        i = np.zeros(0, np.int32); j = np.zeros(0, np.int32)
        sn = np.zeros(0, np.float32); sa = np.zeros(0, np.float32)

    query_time = time.time() - tq
    log(f"Query done: {len(i):,} pairs in {query_time:.1f}s")

    pool_ids = pool["entity_id"].values
    st = blocking_stats(i, pool_ids[j].tolist(), len(s1), gtl, n_pool=len(pool))

    # Oracle ceiling
    from ber.metric import oracle_ceiling
    own = {}
    for k, g in enumerate(gtl):
        for x in g:
            own.setdefault(x, set()).add(k)
    cid = pool_ids[j]
    y = np.fromiter((k in own.get(c, ()) for k, c in zip(i.tolist(), cid)), bool, len(i))
    st["oracle_macro_f05"] = oracle_ceiling(i, y, n_true, len(s1))

    total_time = time.time() - t0
    st["query_seconds"] = query_time
    st["total_seconds"] = total_time
    st["peak_rss_gb"] = rss_gb()
    st["params"] = {"sample_s1": sample_s1, "top_k": top_k, "max_df": max_df}

    log("=" * 60)
    log(f"BLOCKING BENCHMARK  S1={len(s1):,}  pool={len(pool):,}")
    log(f"  pair_recall:          {st.get('pair_recall', 0):.4f}")
    log(f"  entity_recall:        {st.get('macro_entity_recall', 0):.4f}")
    log(f"  full_entity_recall:   {st.get('entities_full_recall', 0):.4f}")
    log(f"  oracle_macro_f05:     {st['oracle_macro_f05']:.4f}")
    log(f"  mean_cands:           {st['mean_cands']:.1f}")
    log(f"  median_cands:         {st['median_cands']:.1f}")
    log(f"  p90_cands:            {st['p90_cands']:.0f}")
    log(f"  p95_cands:            {st['p95_cands']:.0f}")
    log(f"  p99_cands:            {st['p99_cands']:.0f}")
    log(f"  max_cands:            {st['max_cands']}")
    log(f"  zero_cand_frac:       {st['zero_cand_s1_frac']:.4f}")
    log(f"  total_pairs:          {st['total_pairs']:,}")
    log(f"  reduction_ratio:      {st.get('reduction_ratio', 0):.8f}")
    log(f"  query_time:           {query_time:.1f}s")
    log(f"  total_time:           {total_time:.0f}s")
    log(f"  peak_rss:             {st['peak_rss_gb']:.2f} GB")
    log("=" * 60)

    with open(WORK / "blocking_benchmark.jsonl", "a") as f:
        f.write(json.dumps(st, default=float) + "\n")


if __name__ == "__main__":
    main()
