"""End-to-end pipeline CLI (8 GB-safe). Every run appends one JSON line to WORK/experiments.jsonl.

  prepare       --data DATA --work WORK            load TSVs, normalize (v1+v2), cache, GT structure
  renormalize   --work WORK                        add v2 columns to existing v1 pickles (no TSV re-read)
  block-sweep   --work WORK --sample-s1 1000 ...   blocking recall/volume grid (index built once per hash/max_df)
  train         --work WORK --sample-s1 10000 ...  train + validate (+ competitor closure for one-to-one)
  infer         --work WORK --model M --out OUT    streaming test inference with resume
  check-output  --data DATA --out OUT              submission checks
  report        --work WORK                        comparison table of all experiments (markdown)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import decide as D
from . import model as M
from .blocking import Blocker, blocking_stats
from .features import FEATURE_SETS, FeatureContext, pair_features
from .io import read_ground_truth, read_source
from .metric import full_report, oracle_ceiling
from .normalize import normalize_frame, normalize_frame_v2

T0 = time.time()


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.0f}s | peak RSS {rss_gb():.2f} GB] {msg}", flush=True)


def append_log(work, row):
    row = dict(row, peak_rss_gb=rss_gb(), wall_seconds=time.time() - T0, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(Path(work) / "experiments.jsonl", "a") as f:
        f.write(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


# ------------------------------------------------------------------ data
BASE_COLS = ["entity_id", "country", "country_n", "name_clean", "name_core"]
V1_ONLY = ["addr_canon", "postal", "house"]
V1_COLS = BASE_COLS + V1_ONLY
V2_COLS = ["name_key", "name_nonlatin", "addr_canon2", "postal2", "house2", "addr_nums", "addr_cids"]
RAW_COLS = ["business_name", "business_address"]
MAX_RSS_GB = 6.5


def save_slim(df, work, split, s):
    """Write <split>_s<s>.core.pkl (normalized columns) and .raw.pkl (raw text) so later runs
    never load raw text they do not need. The legacy full pickle is kept only if present."""
    core = [c for c in V1_COLS + V2_COLS if c in df.columns]
    df[core].to_pickle(Path(work) / f"{split}_s{s}.core.pkl")
    df[["entity_id"] + RAW_COLS].to_pickle(Path(work) / f"{split}_s{s}.raw.pkl")


def load_df(work, split, s, cols):
    w = Path(work)
    core_p = w / f"{split}_s{s}.core.pkl"
    if not core_p.exists():
        raise SystemExit(f"{core_p.name} missing: run `renormalize` (existing cache) or `prepare`")
    df = pd.read_pickle(core_p)
    need = [c for c in cols if c not in RAW_COLS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"{core_p.name} lacks {missing}: run `renormalize --force`")
    df = df[need]
    if any(c in RAW_COLS for c in cols):
        raw = pd.read_pickle(w / f"{split}_s{s}.raw.pkl")
        assert (raw["entity_id"].values == df["entity_id"].values).all()
        for c in RAW_COLS:
            df[c] = raw[c].values
        del raw
    gc.collect()
    guard(f"loaded {split}_s{s}")
    return df


def guard(stage):
    if rss_gb() > MAX_RSS_GB:
        log(f"STOP: peak RSS {rss_gb():.2f} GB exceeded --max-rss-gb {MAX_RSS_GB} at '{stage}'. "
            "Reduce --sample-s1 / --closure-max / --infer-chunk, or close other applications.")
        sys.exit(3)


def load_split(work, split, cols):
    s1 = load_df(work, split, 1, cols)
    pool = pd.concat([load_df(work, split, 2, cols), load_df(work, split, 3, cols)], ignore_index=True)
    gc.collect()
    return s1, pool


def needed_cols(a, raw=False):
    cols = list(BASE_COLS)
    if a.key_version == 1 or a.feature_set == "v1":
        cols += V1_ONLY
    if a.key_version == 2 or a.feature_set == "v2":
        cols += V2_COLS
    return cols + (RAW_COLS if raw else [])


def pool_owner(work, s1, pool):
    """int32 array: pool row -> S1 row that owns it (-1 = unmatched). Cached. Pure array labels."""
    p = Path(work) / "train_pool_owner.npy"
    if p.exists():
        own = np.load(p)
        if len(own) == len(pool):
            return own
    gt = pd.read_pickle(Path(work) / "train_gt.pkl")
    s1_pos = pd.Index(s1["entity_id"].values)
    pool_pos = pd.Index(pool["entity_id"].values)
    own = np.full(len(pool), -1, np.int32)
    ks = list(gt.keys())
    for a in range(0, len(ks), 200_000):
        part = ks[a:a + 200_000]
        src = np.repeat(s1_pos.get_indexer(part), [len(gt[k]) for k in part])
        dst = pool_pos.get_indexer([x for k in part for x in gt[k]])
        if (src < 0).any() or (dst < 0).any():
            raise SystemExit("ground truth references ids missing from the train files")
        own[dst] = src
    np.save(p, own)
    del gt
    gc.collect()
    return own


def gt_groups(own, rows):
    """For S1 rows `rows` (global indices), list of arrays of owned pool rows."""
    o = np.argsort(own, kind="stable")
    so = own[o]
    lo = np.searchsorted(so, rows, "left")
    hi = np.searchsorted(so, rows, "right")
    return [o[a:b] for a, b in zip(lo, hi)]


def sample_rows(s1_df, n, seed):
    """Global row positions of the sampled S1 — IDENTICAL to the EXP-000/EXP-001 baseline sampler
    (pandas DataFrame.sample(n, random_state=seed)), so sweeps reuse the same benchmark S1."""
    if not n or n >= len(s1_df):
        return np.arange(len(s1_df)) if len(s1_df) == len(s1_df.index) and s1_df.index.equals(pd.RangeIndex(len(s1_df))) \
            else np.asarray(s1_df.index)
    return np.asarray(s1_df.sample(n=n, random_state=seed).index)


def split_role(ids, salt, val_pct=20, es_pct=10):
    b = np.array([int(hashlib.md5((salt + x).encode()).hexdigest()[:8], 16) % 100 for x in ids])
    return np.where(b < val_pct, "val", np.where(b < val_pct + es_pct, "es", "fit"))


# ------------------------------------------------------------------ prepare / renormalize
def cmd_prepare(a):
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            p = Path(a.data) / split / f"{split}_source{s}.tsv"
            df = read_source(p)
            bad = ~df["entity_id"].str.startswith(f"S{s}-")
            if bad.any():
                raise SystemExit(f"{p}: {int(bad.sum())} ids without prefix S{s}-")
            normalize_frame(df)
            normalize_frame_v2(df)
            save_slim(df, work, split, s)
            log(f"prepared {p.name}: {len(df):,} rows; countries={df['country'].value_counts().to_dict()}; "
                f"empty addr={(df['business_address'] == '').mean() * 100:.2f}%; "
                f"non-Latin names={df['name_nonlatin'].mean() * 100:.2f}%")
            del df
            gc.collect()
    gt = read_ground_truth(Path(a.data) / "train" / "train_ground_truth.tsv")
    pd.to_pickle(gt, work / "train_gt.pkl")
    _gt_structure(work, gt)


def _gt_structure(work, gt):
    cnt = {}
    for v in gt.values():
        for x in v:
            cnt[x] = cnt.get(x, 0) + 1
    by_src = {}
    for x, c in cnt.items():
        s = x[:2]
        d = by_src.setdefault(s, {"ids": 0, "ids_in_multiple_s1": 0})
        d["ids"] += 1
        d["ids_in_multiple_s1"] += c > 1
    n_match = [len(v) for v in gt.values()]
    info = {"n_s1": len(gt), "n_ids": len(cnt), "ids_in_multiple_s1": sum(c > 1 for c in cnt.values()),
            "by_source": by_src, "singleton_frac": float(np.mean([n == 0 for n in n_match])),
            "mean_matches": float(np.mean(n_match)), "max_matches": int(max(n_match))}
    info["one_to_one"] = info["ids_in_multiple_s1"] == 0
    (Path(work) / "gt_structure.json").write_text(json.dumps(info, indent=2))
    log(f"GT structure: {info}")


def cmd_renormalize(a):
    """Upgrade an existing cache (legacy <split>_s<s>.pkl from the v1 pipeline) to slim v2 files."""
    w = Path(a.work)
    for split in ("train", "test"):
        for s in (1, 2, 3):
            core_p = w / f"{split}_s{s}.core.pkl"
            if core_p.exists() and not a.force:
                log(f"{core_p.name} exists, skipped")
                continue
            legacy = w / f"{split}_s{s}.pkl"
            if legacy.exists():
                df = pd.read_pickle(legacy)
            else:
                df = pd.read_pickle(core_p)
                raw = pd.read_pickle(w / f"{split}_s{s}.raw.pkl")
                for c in RAW_COLS:
                    df[c] = raw[c].values
                del raw
            if "name_core" not in df:
                normalize_frame(df)
            normalize_frame_v2(df)
            save_slim(df, w, split, s)
            log(f"{split}_s{s}: slim v2 cache written ({len(df):,} rows, non-Latin names "
                f"{df['name_nonlatin'].mean() * 100:.2f}%); legacy pickle can be deleted after verification")
            del df
            gc.collect()
    if not (w / "gt_structure.json").exists() or a.force:
        _gt_structure(w, pd.read_pickle(w / "train_gt.pkl"))


# ------------------------------------------------------------------ blocking sweep
def make_blocker(a, hash_bits=None, max_df=None, top_k=None):
    fam = a.families.split(",") if a.families else None
    return Blocker(hash_bits=hash_bits or a.hash_bits, max_df=max_df or a.max_df, top_k=top_k or a.top_k,
                   min_score=a.min_score, key_version=a.key_version, families=fam, spill_dir=a.work)


def run_blocking(blk, s1_df, chunk=5000):
    I, J, SN, SA, FB = [], [], [], [], []
    for i, j, sn, sa, fb in blk.query(s1_df, chunk=chunk):
        I.append(i); J.append(j); SN.append(sn); SA.append(sa); FB.append(fb)
    if not I:
        z = np.zeros(0, np.int32)
        return z, z, np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.uint8)
    i, j, sn, sa, fb = (np.concatenate(x) for x in (I, J, SN, SA, FB))
    o = np.lexsort((-np.maximum(sn, sa), i))
    return i[o], j[o], sn[o], sa[o], fb[o]


def cmd_block_sweep(a):
    s1_all, pool = load_split(a.work, "train", needed_cols(a, raw=a.dump_misses))
    own = pool_owner(a.work, s1_all, pool)
    rows = sample_rows(s1_all, a.sample_s1, a.seed)
    s1 = s1_all.iloc[rows].reset_index(drop=True)
    gtl = gt_groups(own, rows)
    n_true = np.array([len(g) for g in gtl], float)
    log(f"sweep sample: {len(s1):,} S1 (seed {a.seed}) vs FULL pool {len(pool):,}")
    for hb in [int(x) for x in a.hash_bits_list.split(",")]:
        for mdf in [int(x) for x in a.max_df_list.split(",")]:
            blk = make_blocker(a, hash_bits=hb, max_df=mdf)
            t = time.time()
            blk.fit_pool(pool, log)
            guard("index build")
            build_s = time.time() - t
            for tk in [int(x) for x in a.top_k_list.split(",")]:
                blk.set_top_k(tk)
                t = time.time()
                i, j, sn, sa, fb = run_blocking(blk, s1)
                q_s = time.time() - t
                st, hit = blocking_stats(i, j, len(s1), gtl, fb, n_pool=len(pool))
                y = own[j] == rows[i]
                st["oracle_macro_f05"] = oracle_ceiling(i, y, n_true, len(s1))
                exp = f"{a.exp_id}-v{a.key_version}-h{hb}-df{mdf}-k{tk}"
                rec = {"exp": exp, "type": "block", "sample_s1": len(s1), "seed": a.seed,
                       "key_version": a.key_version, "families": a.families or "default", "hash_bits": hb,
                       "max_df": mdf, "top_k": tk, **st, "index_build_s": build_s, "query_s": q_s,
                       "query_s_per_s1": q_s / max(len(s1), 1), "index": blk.index.stats}
                append_log(a.work, rec)
                log(f"{exp}: pair_recall={st['pair_recall']:.4f} full_ent={st['entities_full_recall']:.4f} "
                    f"oracle={st['oracle_macro_f05']:.4f} mean={st['mean_cands']:.1f} p95={st['p95_cands']:.0f} "
                    f"addr_only={st.get('pair_recall_addr_only', 0):.4f} q={q_s:.1f}s")
                if a.dump_misses:
                    gi = np.repeat(np.arange(len(s1)), n_true.astype(int))
                    gj = np.concatenate(gtl) if len(gtl) else np.zeros(0, int)
                    mm = ~hit
                    pd.DataFrame({"s1": s1["entity_id"].values[gi[mm]], "cand": pool["entity_id"].values[gj[mm]],
                                  "s1_name": s1["business_name"].values[gi[mm]], "cand_name": pool["business_name"].values[gj[mm]],
                                  "s1_addr": s1["business_address"].values[gi[mm]], "cand_addr": pool["business_address"].values[gj[mm]],
                                  }).to_csv(Path(a.work) / f"misses_{exp}.tsv", sep="\t", index=False)
            del blk
            gc.collect()


# ------------------------------------------------------------------ train / validate
def cmd_train(a):
    work = Path(a.work)
    names = FEATURE_SETS[a.feature_set]
    s1_all, pool = load_split(work, "train", needed_cols(a, raw=a.dump_val_errors))
    own = pool_owner(work, s1_all, pool)
    rng = np.random.default_rng(a.seed)
    pool_s1 = s1_all if not a.countries else s1_all[s1_all["country"].isin(a.countries).values]
    rows = sample_rows(pool_s1, a.sample_s1, a.seed)
    role = split_role(s1_all["entity_id"].values[rows], a.split_salt)
    if a.val_countries:  # cross-country transfer: validate only on these countries, never train on them
        c = s1_all["country"].values[rows]
        isv = np.isin(c, a.val_countries)
        role = np.where(isv, "val", np.where(role == "val", "fit", role))
        if a.train_countries:
            role = np.where(~isv & ~np.isin(c, a.train_countries), "drop", role)
    blk = make_blocker(a)
    blk.fit_pool(pool, log)
    comp_rows = np.zeros(0, np.int64)
    if a.closure:
        # competitor closure: every other S1 that could claim a val S1's candidates at test time
        vrows = rows[role == "val"]
        vi, vj, *_ = run_blocking(blk, s1_all.iloc[vrows].reset_index(drop=True))
        owners = np.unique(own[vj])
        owners = owners[owners >= 0]
        key = s1_all["country_n"].values + "|" + s1_all["name_key"].values if "name_key" in s1_all else None
        same = np.zeros(0, np.int64)
        if key is not None:
            vk = set(key[vrows])
            same = np.flatnonzero(pd.Series(key).isin(vk).values)
            if len(same) > a.closure_max // 2:
                same = rng.choice(same, a.closure_max // 2, replace=False)
        comp_rows = np.setdiff1d(np.union1d(owners, same), vrows)
        if len(comp_rows) > a.closure_max:
            log(f"WARNING closure capped {len(comp_rows):,} -> {a.closure_max:,}")
            comp_rows = rng.choice(comp_rows, a.closure_max, replace=False)
        in_sample = np.isin(rows, comp_rows)
        moved = int((in_sample & (role != "val")).sum())
        if moved > 0.10 * max(int((role != "val").sum()), 1):
            log(f"WARNING closure removes {moved:,} S1 from training (>10%): sample is a large share of all S1")
        role = np.where(in_sample & (role != "val"), "comp", role)   # competitors are never trained on
        comp_rows = np.setdiff1d(comp_rows, rows)
        log(f"competitor closure: {len(comp_rows):,} extra S1 + {int(in_sample.sum()):,} sampled S1 moved to 'comp'")
    all_rows = np.concatenate([rows, comp_rows])
    all_role = np.concatenate([role, np.full(len(comp_rows), "comp")])
    s1 = s1_all.iloc[all_rows].reset_index(drop=True)
    gtl = gt_groups(own, all_rows)
    del s1_all
    gc.collect()
    log(f"S1 roles: {pd.Series(all_role).value_counts().to_dict()}  pool={len(pool):,}")
    ctx = FeatureContext(s1, pool, a.feature_set)
    t = time.time()
    I, J, X, Y = [], [], [], []
    for i, j, sn, sa, fb in blk.query(s1, chunk=a.chunk):
        o = np.lexsort((-np.maximum(sn, sa), i))
        i, j, sn, sa, fb = i[o], j[o], sn[o], sa[o], fb[o]
        X.append(pair_features(ctx, i, j, sn, sa, fb)); I.append(i); J.append(j)
        Y.append(own[j] == all_rows[i])
        guard("training features")
    i, j, X, y = np.concatenate(I), np.concatenate(J), np.vstack(X), np.concatenate(Y)
    feat_s = time.time() - t
    r = all_role[i]
    log(f"pairs={len(i):,} pos={int(y.sum()):,} neg={int((~y).sum()):,} pos_rate={y.mean():.4f} "
        f"features={X.shape[1]} ({feat_s:.0f}s)")
    t = time.time()
    model = M.train(X[r == "fit"], y[r == "fit"], X[r == "es"], y[r == "es"], names, a.model, a.scale_pos_weight)
    train_s = time.time() - t

    # ---- validation (+ competitors) : official macro F0.5 over ALL val S1
    ev_rows = np.flatnonzero(all_role == "val")
    cp_rows = np.flatnonzero(all_role == "comp")
    remap = np.full(len(s1), -1)
    remap[ev_rows] = np.arange(len(ev_rows))
    remap[cp_rows] = len(ev_rows) + np.arange(len(cp_rows))
    m = remap[i] >= 0
    g, cand, yv = remap[i[m]], j[m], y[m]
    prob = M.predict(model, X[m])
    is_s3 = X[m][:, names.index("is_s3")]
    n_true = np.array([len(gtl[k]) for k in ev_rows], float)
    n_groups = len(ev_rows) + len(cp_rows)
    info = json.loads((work / "gt_structure.json").read_text())
    best = D.tune(g, cand, prob, is_s3, yv, n_true, n_groups, allow_o2o=info["one_to_one"], n_eval=len(ev_rows))
    keep = D.decide(g, cand, prob, is_s3, n_groups, best["t2"], best["t3"], best["rel"], best["o2o"])
    ev = g < len(ev_rows)
    rep, pe = full_report(g[ev], keep[ev], yv[ev], n_true, len(ev_rows))
    ceiling = oracle_ceiling(g[ev], yv[ev], n_true, len(ev_rows))
    # slices
    vs1 = s1.iloc[ev_rows]
    slices = {f"country={c}": float(pe[(vs1["country"] == c).values].mean()) for c in vs1["country"].unique()}
    if "name_nonlatin" in pool:
        nl = np.array([pool["name_nonlatin"].values[gtl[k]].any() if len(gtl[k]) else False for k in ev_rows])
        slices["has_nonlatin_true_match"] = float(pe[nl].mean()) if nl.any() else None
    acol = "addr_canon2" if "addr_canon2" in pool else "addr_canon"
    ea = np.array([(pool[acol].values[gtl[k]] == "").any() if len(gtl[k]) else False for k in ev_rows])
    slices["has_empty_addr_true_match"] = float(pe[ea].mean()) if ea.any() else None
    slices["singletons"] = rep["singleton_f05"]
    slices["multi_match"] = rep["multi_match_f05"]
    res = {"exp": a.exp_id, "type": "train", "sample_s1": int(len(rows)), "seed": a.seed, "split_salt": a.split_salt,
           "key_version": a.key_version, "families": a.families or "default", "hash_bits": a.hash_bits,
           "max_df": a.max_df, "top_k": a.top_k, "min_score": a.min_score, "feature_set": a.feature_set,
           "model": a.model, "scale_pos_weight": a.scale_pos_weight, "closure": a.closure,
           "n_val_s1": int(len(ev_rows)), "n_comp_s1": int(len(cp_rows)),
           "val_macro_f05": rep["macro_f05"], "blocking_ceiling_f05": ceiling,
           "blocking_pair_recall": float(yv[ev].sum() / max(n_true.sum(), 1)),
           **rep, "decision": best, "slices": slices,
           "n_pairs": int(len(i)), "n_pos": int(y.sum()), "n_neg": int((~y).sum()), "pos_rate": float(y.mean()),
           "mean_cands": float(len(i) / len(s1)), "feature_seconds": feat_s, "train_seconds": train_s,
           "model_iterations": int(getattr(model, "best_iteration", 0) or 0),
           "feature_importance_top15": dict(list(M.feature_importance(model, names).items())[:15])}
    append_log(work, res)
    log("RESULT " + json.dumps({k: res[k] for k in ("exp", "val_macro_f05", "blocking_ceiling_f05", "blocking_pair_recall",
                                                   "micro_precision", "micro_recall", "singleton_accuracy",
                                                   "multi_match_f05", "decision")}, default=str, indent=1))
    mp = work / "models" / f"{a.exp_id}.txt"
    M.save(model, mp, {"feature_set": a.feature_set, "feature_names": names, "decision": best,
                       "blocker": {"key_version": a.key_version, "families": a.families, "hash_bits": a.hash_bits,
                                   "max_df": a.max_df, "top_k": a.top_k, "min_score": a.min_score},
                       "validation": {k: res[k] for k in ("val_macro_f05", "blocking_ceiling_f05", "n_val_s1")}})
    log(f"saved model -> {mp}")
    if a.dump_val_errors:
        from .errors import dump_and_summarize
        dump_and_summarize(work / f"val_errors_{a.exp_id}.tsv", s1, pool, ev_rows, g[ev], cand[ev], prob[ev],
                           keep[ev], yv[ev], gtl, X[m][ev], names, pe, log)


# ------------------------------------------------------------------ infer (streaming, resumable)
def cmd_infer(a):
    work = Path(a.work)
    if not a.model.endswith(".txt"):
        raise SystemExit("infer needs --model work/models/<EXP>.txt")
    model, meta = M.load(a.model, FEATURE_SETS[meta_fs(a.model)])
    dec, bp = meta["decision"], meta["blocker"]
    a.key_version, a.families, a.feature_set = bp["key_version"], bp["families"], meta["feature_set"]
    names = FEATURE_SETS[a.feature_set]
    s1, pool = load_split(work, "test", needed_cols(a))
    if a.sample_s1:
        s1 = s1.sample(n=min(a.sample_s1, len(s1)), random_state=42).reset_index(drop=True)
        log("TIMING REHEARSAL on an S1 subset: outputs are PARTIAL and NOT a valid submission")
    log(f"test S1={len(s1):,} pool={len(pool):,} countries={s1['country'].value_counts().to_dict()}")
    blk = Blocker(hash_bits=bp["hash_bits"], max_df=bp["max_df"], top_k=bp["top_k"], min_score=bp["min_score"],
                  key_version=bp["key_version"], families=bp["families"].split(",") if bp["families"] else None,
                  spill_dir=a.work).fit_pool(pool, log)
    ctx = FeatureContext(s1, pool, a.feature_set)
    cdir = work / ("infer_chunks_rehearsal" if a.sample_s1 else "infer_chunks")
    if a.fresh and cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(exist_ok=True)
    t = time.time()
    n_done = 0
    for c0 in range(0, len(s1), a.infer_chunk):
        f = cdir / f"chunk_{c0:09d}.npz"
        if f.exists():
            continue                                      # resume
        sub = s1.iloc[c0:c0 + a.infer_chunk]
        I, J, P, S3 = [], [], [], []
        for i, j, sn, sa, fb in blk.query(sub, chunk=a.chunk):
            o = np.lexsort((-np.maximum(sn, sa), i))
            i, j, sn, sa, fb = i[o], j[o], sn[o], sa[o], fb[o]
            gi = i + c0
            X = pair_features(ctx, gi, j, sn, sa, fb)
            P.append(M.predict(model, X)); I.append(gi.astype(np.int32)); J.append(j)
            S3.append(X[:, names.index("is_s3")] > 0.5)
        cat = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)
        guard("inference chunk")
        np.savez(f, i=cat(I, np.int32), j=cat(J, np.int32), p=cat(P, np.float32), s3=cat(S3, bool))
        n_done += len(sub)
        el = time.time() - t
        log(f"chunk {c0:,}: {len(sub):,} S1 scored; throughput {n_done / max(el, 1e-9):.1f} S1/s; "
            f"ETA {(len(s1) - c0 - len(sub)) / max(n_done / max(el, 1e-9), 1e-9) / 3600:.2f} h")
    # ---- global decision (one-to-one needs every S1's scores for the same S2/S3 record)
    del blk, ctx
    gc.collect()
    files = sorted(cdir.glob("chunk_*.npz"))
    d = [np.load(f) for f in files]
    i = np.concatenate([x["i"] for x in d]); j = np.concatenate([x["j"] for x in d])
    p = np.concatenate([x["p"] for x in d]); s3 = np.concatenate([x["s3"] for x in d])
    del d
    keep = D.decide(i, j, p, s3.astype(np.float32), len(s1), dec["t2"], dec["t3"], dec["rel"], dec["o2o"])
    log(f"decision: {len(i):,} candidate pairs -> {int(keep.sum()):,} matches (o2o={dec['o2o']})")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids, pid = s1["entity_id"].values, pool["entity_id"].values
    bounds = np.searchsorted(i, np.arange(len(s1) + 1))       # i is sorted: chunks in order, sorted inside
    assert np.all(np.diff(i) >= 0), "pairs must be grouped by S1"
    with open(out / "candidate_pairs.tsv", "w", encoding="utf-8") as fc, \
            open(out / "matching_results.tsv", "w", encoding="utf-8") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for k in range(len(s1)):
            a0, a1 = bounds[k], bounds[k + 1]
            fc.write(f"{ids[k]}\t{','.join(pid[j[a0:a1]])}\n")
            fm.write(f"{ids[k]}\t{','.join(pid[j[a0:a1][keep[a0:a1]]])}\n")
    npred = np.bincount(i[keep], minlength=len(s1))
    stats = {"exp": "INFER" + ("-rehearsal" if a.sample_s1 else ""), "type": "infer", "model": str(a.model),
             "n_s1": len(s1), "pairs": int(len(i)), "matches": int(keep.sum()),
             "mean_cands": float(len(i) / len(s1)), "no_match_s1": int((npred == 0).sum()),
             "seconds_scoring": time.time() - t,
             "by_country": {c: {"mean_matches": float(npred[(s1['country'] == c).values].mean()),
                                "empty_frac": float((npred[(s1['country'] == c).values] == 0).mean()),
                                "mean_cands": float(np.bincount(i, minlength=len(s1))[(s1['country'] == c).values].mean())}
                            for c in s1["country"].unique()}}
    append_log(work, stats)
    log("INFER " + json.dumps(stats, indent=1, default=str))


def meta_fs(model_path):
    return json.loads(Path(model_path).with_suffix(".json").read_text())["feature_set"]


# ------------------------------------------------------------------ misc
def cmd_check_output(a):
    from .check_output import check
    sys.exit(0 if check(Path(a.out) / "matching_results.tsv", Path(a.out) / "candidate_pairs.tsv",
                        Path(a.data) / "test") else 1)


def cmd_report(a):
    rows = [json.loads(l) for l in open(Path(a.work) / "experiments.jsonl")]
    b = [r for r in rows if r.get("type", "block" if "pair_recall" in r and "val_macro_f05" not in r else "train") == "block"
         or ("pair_recall" in r and "val_macro_f05" not in r)]
    t = [r for r in rows if "val_macro_f05" in r]
    L = ["## Blocking experiments", "",
         "| exp | pair recall | full-recall ent. | oracle F0.5 | mean | p95 | p99 | addr-only recall | query s/S1 | build s | peak RSS GB |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in b:
        L.append(f"| {r.get('exp', 'baseline')} | {r['pair_recall']:.4f} | {r['entities_full_recall']:.4f} | "
                 f"{r.get('oracle_macro_f05', float('nan')):.4f} | {r['mean_cands']:.1f} | {r['p95_cands']:.0f} | "
                 f"{r.get('p99_cands', float('nan')):.0f} | {r.get('pair_recall_addr_only', float('nan')):.4f} | "
                 f"{r.get('query_s_per_s1', r.get('query_seconds', float('nan')) / max(r.get('n_s1', 1), 1)):.4f} | "
                 f"{r.get('index_build_s', float('nan')):.0f} | {r.get('peak_rss_gb', float('nan')):.2f} |")
    L += ["", "## Model / decision experiments", "",
          "| exp | val S1 | macro F0.5 | ceiling | block recall | micro P | micro R | singleton acc | multi F0.5 | decision | peak RSS GB |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t:
        d = r.get("decision", {})
        L.append(f"| {r['exp']} | {r.get('n_val_s1', r.get('val_s1'))} | {r['val_macro_f05']:.4f} | {r['blocking_ceiling_f05']:.4f} | "
                 f"{r.get('blocking_pair_recall', float('nan')):.4f} | {r.get('micro_precision', r.get('pair_precision', float('nan'))):.4f} | "
                 f"{r.get('micro_recall', r.get('pair_recall', float('nan'))):.4f} | {r.get('singleton_accuracy') or float('nan'):.4f} | "
                 f"{r.get('multi_match_f05') or float('nan'):.4f} | t={d.get('t2')}/{d.get('t3')} rel={d.get('rel')} o2o={d.get('o2o')} | "
                 f"{r.get('peak_rss_gb', float('nan')):.2f} |")
    txt = "\n".join(L)
    (Path(a.work) / "EXPERIMENTS.md").write_text(txt)
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("prepare", cmd_prepare), ("renormalize", cmd_renormalize), ("block-sweep", cmd_block_sweep),
                     ("train", cmd_train), ("infer", cmd_infer), ("check-output", cmd_check_output),
                     ("report", cmd_report)]:
        p = sub.add_parser(name)
        p.set_defaults(func=fn)
        p.add_argument("--work", default="work")
        p.add_argument("--data", default="dataset")
        p.add_argument("--out", default="output")
        p.add_argument("--exp-id", default="EXP")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--sample-s1", type=int, default=0)
        p.add_argument("--key-version", type=int, default=2)
        p.add_argument("--families", default="", help="comma list of v2 key families (ablation); empty = all")
        p.add_argument("--hash-bits", type=int, default=24)
        p.add_argument("--max-df", type=int, default=5000)
        p.add_argument("--top-k", type=int, default=40)
        p.add_argument("--min-score", type=float, default=0.05)
        p.add_argument("--hash-bits-list", default="24")
        p.add_argument("--max-df-list", default="5000")
        p.add_argument("--top-k-list", default="40")
        p.add_argument("--dump-misses", action="store_true")
        p.add_argument("--feature-set", default="v2", choices=list(FEATURE_SETS))
        p.add_argument("--model", default="lgbm", help="lgbm | lr (train); path to model .txt (infer)")
        p.add_argument("--scale-pos-weight", type=float, default=1.0)
        p.add_argument("--split-salt", default="")
        p.add_argument("--closure", action="store_true", help="competitor closure for one-to-one validation")
        p.add_argument("--closure-max", type=int, default=200_000)
        p.add_argument("--countries", nargs="*", default=None)
        p.add_argument("--train-countries", nargs="*", default=None)
        p.add_argument("--val-countries", nargs="*", default=None)
        p.add_argument("--dump-val-errors", action="store_true")
        p.add_argument("--chunk", type=int, default=2000, help="S1 rows per blocking/feature batch")
        p.add_argument("--infer-chunk", type=int, default=20000, help="S1 rows per resumable inference file")
        p.add_argument("--fresh", action="store_true", help="infer: discard previous chunk files")
        p.add_argument("--force", action="store_true")
        p.add_argument("--max-rss-gb", type=float, default=6.5, help="stop cleanly above this peak RSS")
    a = ap.parse_args(argv)
    global MAX_RSS_GB
    MAX_RSS_GB = a.max_rss_gb
    a.func(a)


if __name__ == "__main__":
    main()
