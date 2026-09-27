"""Memory-efficient candidate generation: compact CSR-style inverted index.

Previous blocker built two transposed sparse matrices (10.3M × 16M) ≈ 4 GB RAM.
This version uses a CSR-style inverted index: flat numpy arrays with offset delimiters.

Two-pass construction keeps peak memory minimal:
  Pass 1: count document frequencies (just a numpy counter array)
  Pass 2: fill pre-allocated flat arrays (no Python lists/dicts at scale)

Estimated index memory for 10.3M pool with max_df=5000: ~400 MB total
(vs ~4 GB for old sparse approach).

Key design:
  - Single shared index with two weight channels (name & addr)
  - CSR offsets into flat doc_id and weight arrays
  - Per-query numpy-vectorized scoring via concatenate + unique + bincount
  - Country scoping via key prefixes (open-set, no hard-coded countries)
"""
from __future__ import annotations

import math
import sys
import time
import zlib
from typing import Dict, Iterator, List, Tuple

import numpy as np

PROFILES = {
    "name": {"n": 1.0, "p": 0.35, "c": 1.2, "a": 0.15, "z": 0.2, "nz": 0.8, "nh": 0.6},
    "addr": {"n": 0.3, "p": 0.15, "c": 0.3, "a": 1.0, "z": 1.0, "nz": 1.0, "nh": 1.0},
}

_DEFAULT_N_FEATURES = 2 ** 20  # 1M buckets


def record_keys(country, core, addr, postal, house, FIELD_W):
    """Generate (string_key, weight) pairs for a record under a given profile."""
    c = country + "|"
    toks = core.split()
    out = []
    for t in toks:
        out.append((c + "n:" + t, FIELD_W["n"]))
        if len(t) >= 5:
            out.append((c + "p:" + t[:4], FIELD_W["p"]))
    if toks:
        out.append((c + "c:" + "".join(sorted(toks)), FIELD_W["c"]))
        t1 = toks[0]
        if postal:
            out.append((c + "nz:" + t1 + "|" + postal, FIELD_W["nz"]))
        if house:
            out.append((c + "nh:" + t1 + "|" + house, FIELD_W["nh"]))
    for t in set(addr.split()):
        out.append((c + "a:" + t, FIELD_W["a"]))
    if postal:
        out.append((c + "z:" + postal, FIELD_W["z"]))
    return out


def _hash_key(key: str, n_features: int) -> int:
    """Deterministic hash via CRC32 (C-implemented, stable across runs)."""
    return zlib.crc32(key.encode('utf-8')) % n_features


class CompactIndex:
    """CSR-style inverted index with two weight channels.

    Memory layout (all flat numpy arrays):
      offsets[n_features+1]  int64   — posting list delimiters
      doc_ids[N]             int32   — pool row indices
      w_name[N]              float32 — IDF*field_weight for name channel
      w_addr[N]              float32 — IDF*field_weight for addr channel
      doc_norms_name[n_pool] float32 — per-doc L2 norms (name channel)
      doc_norms_addr[n_pool] float32 — per-doc L2 norms (addr channel)
      idf[n_features]        float32 — IDF values

    N = total kept postings (keys with df <= max_df).
    """

    def __init__(self, n_features: int, max_df: int):
        self.n_features = n_features
        self.max_df = max_df

    def build(self, pool_df):
        t0 = time.time()
        n = len(pool_df)
        self.n_docs = n

        countries = pool_df["country_n"].values
        cores = pool_df["name_core"].values
        addrs = pool_df["addr_canon"].values
        postals = pool_df["postal"].values
        houses = pool_df["house"].values

        nf = self.n_features
        report = max(1, n // 10)

        # ---- Pass 1: count document frequencies ----
        df_count = np.zeros(nf, dtype=np.int32)
        print(f"  [pass1] df counting {n:,} records ...", flush=True)
        for ri in range(n):
            keys = record_keys(countries[ri], cores[ri], addrs[ri],
                               postals[ri], houses[ri], PROFILES["name"])
            seen = set()
            for ks, _ in keys:
                h = _hash_key(ks, nf)
                if h not in seen:
                    seen.add(h)
                    df_count[h] += 1
            if (ri + 1) % report == 0:
                print(f"    {ri+1:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)

        # ---- Compute IDF, offsets ----
        self.idf = np.log((n + 1) / (df_count + 1)).astype(np.float32) + 1.0
        self.idf[(df_count > self.max_df) | (df_count == 0)] = 0.0
        kept_df = np.where(self.idf > 0, df_count, 0).astype(np.int64)
        self.offsets = np.zeros(nf + 1, dtype=np.int64)
        np.cumsum(kept_df, out=self.offsets[1:])
        total = int(self.offsets[-1])
        n_dropped = int((df_count > self.max_df).sum())
        n_kept = int((kept_df > 0).sum())
        print(f"  [pass1 done] {n_kept:,} keys kept, {n_dropped:,} dropped, "
              f"{total:,} postings ({total*12/1024**2:.0f} MB), {time.time()-t0:.0f}s", flush=True)

        # ---- Pass 2: fill posting lists ----
        self.doc_ids = np.empty(total, dtype=np.int32)
        self.w_name = np.empty(total, dtype=np.float32)
        self.w_addr = np.empty(total, dtype=np.float32)
        cursors = self.offsets[:-1].copy()
        norm2_n = np.zeros(n, dtype=np.float64)
        norm2_a = np.zeros(n, dtype=np.float64)

        t1 = time.time()
        print(f"  [pass2] filling posting lists ...", flush=True)
        idf = self.idf
        wn_prof, wa_prof = PROFILES["name"], PROFILES["addr"]
        for ri in range(n):
            keys_n = record_keys(countries[ri], cores[ri], addrs[ri],
                                 postals[ri], houses[ri], wn_prof)
            keys_a = record_keys(countries[ri], cores[ri], addrs[ri],
                                 postals[ri], houses[ri], wa_prof)
            seen = set()
            for (ks, wn), (_, wa) in zip(keys_n, keys_a):
                h = _hash_key(ks, nf)
                if h in seen:
                    continue
                seen.add(h)
                idf_h = idf[h]
                if idf_h == 0.0:
                    continue
                pos = int(cursors[h])
                self.doc_ids[pos] = ri
                wn_val = wn * idf_h
                wa_val = wa * idf_h
                self.w_name[pos] = wn_val
                self.w_addr[pos] = wa_val
                cursors[h] = pos + 1
                norm2_n[ri] += wn_val * wn_val
                norm2_a[ri] += wa_val * wa_val
            if (ri + 1) % report == 0:
                print(f"    {ri+1:,}/{n:,} ({time.time()-t1:.0f}s)", flush=True)

        del cursors
        self.doc_norms_n = np.sqrt(norm2_n).astype(np.float32)
        self.doc_norms_a = np.sqrt(norm2_a).astype(np.float32)
        self.doc_norms_n[self.doc_norms_n == 0] = 1.0
        self.doc_norms_a[self.doc_norms_a == 0] = 1.0
        del norm2_n, norm2_a

        mem = (total * 12 + n * 8 + nf * 12) / 1024**2
        print(f"  [index done] ~{mem:.0f} MB, {time.time()-t0:.0f}s total", flush=True)
        return self

    def query_one(self, keys, channel, top_k, min_score):
        """Query for one record. Returns (pool_ids int32, cosine_scores float32)."""
        nf = self.n_features
        idf = self.idf
        q_terms: Dict[int, float] = {}
        for ks, w in keys:
            h = _hash_key(ks, nf)
            idf_h = idf[h]
            if idf_h == 0.0:
                continue
            qv = w * idf_h
            q_terms[h] = q_terms.get(h, 0.0) + qv

        if not q_terms:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)

        q_norm = math.sqrt(sum(v * v for v in q_terms.values()))
        if q_norm < 1e-12:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)

        w_arr = self.w_name if channel == "name" else self.w_addr
        offsets = self.offsets
        doc_ids = self.doc_ids

        # Gather posting lists
        id_parts, contrib_parts = [], []
        for h, qv in q_terms.items():
            s, e = int(offsets[h]), int(offsets[h + 1])
            if s == e:
                continue
            id_parts.append(doc_ids[s:e])
            contrib_parts.append(w_arr[s:e] * qv)

        if not id_parts:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)

        cat_ids = np.concatenate(id_parts)
        cat_w = np.concatenate(contrib_parts).astype(np.float64)

        # Aggregate per-document scores
        uniq, inv = np.unique(cat_ids, return_inverse=True)
        raw = np.bincount(inv, weights=cat_w, minlength=len(uniq)).astype(np.float32)

        # Cosine normalisation
        dnorms = self.doc_norms_n if channel == "name" else self.doc_norms_a
        cos = raw / (q_norm * dnorms[uniq])

        # Min-score + top-k
        mask = cos >= min_score
        uniq, cos = uniq[mask], cos[mask]
        if len(uniq) > top_k:
            tk = np.argpartition(-cos, top_k)[:top_k]
            uniq, cos = uniq[tk], cos[tk]

        return uniq.astype(np.int32), cos.astype(np.float32)


class Blocker:
    """Union of name-channel top-k and addr-channel top-k per S1 query.
    Uses CompactIndex — memory-safe on 8 GB Mac."""

    def __init__(self, n_features=_DEFAULT_N_FEATURES, max_df=5000, top_k=40,
                 min_score=0.05, addr_share=0.4, nnz_budget=20_000_000):
        self.n_features = n_features
        self.max_df = max_df
        self.min_score = min_score
        self.nnz_budget = nnz_budget
        self.k_addr = max(1, int(round(top_k * addr_share)))
        self.k_name = max(1, top_k - self.k_addr)
        self.top_k = top_k

    def fit_pool(self, pool_df):
        t = time.time()
        self.n_pool = len(pool_df)
        print(f"[blocking] building index pool={self.n_pool:,} nf={self.n_features:,} "
              f"max_df={self.max_df:,}")
        self.index = CompactIndex(self.n_features, self.max_df)
        self.index.build(pool_df)
        print(f"[blocking] k_name={self.k_name} k_addr={self.k_addr} ({time.time()-t:.1f}s)")
        return self

    def query(self, s1_df, chunk=5000) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (s1_row_idx, pool_idx, score_name, score_addr) per chunk."""
        n_s1 = len(s1_df)
        countries = s1_df["country_n"].values
        cores = s1_df["name_core"].values
        addrs_v = s1_df["addr_canon"].values
        postals = s1_df["postal"].values
        houses_v = s1_df["house"].values
        idx = self.index
        wn_prof, wa_prof = PROFILES["name"], PROFILES["addr"]

        for start in range(0, n_s1, chunk):
            stop = min(start + chunk, n_s1)
            t0 = time.time()
            out_r, out_c, out_sn, out_sa = [], [], [], []

            for q in range(start, stop):
                c, co, ad, po, ho = (countries[q], cores[q], addrs_v[q],
                                     postals[q], houses_v[q])
                keys_n = record_keys(c, co, ad, po, ho, wn_prof)
                keys_a = record_keys(c, co, ad, po, ho, wa_prof)

                nids, nsc = idx.query_one(keys_n, "name", self.k_name, self.min_score)
                aids, asc = idx.query_one(keys_a, "addr", self.k_addr, self.min_score)

                # Union merge
                pairs: Dict[int, List[float]] = {}
                for k in range(len(nids)):
                    pairs[int(nids[k])] = [float(nsc[k]), 0.0]
                for k in range(len(aids)):
                    pid = int(aids[k])
                    if pid in pairs:
                        pairs[pid][1] = float(asc[k])
                    else:
                        pairs[pid] = [0.0, float(asc[k])]

                for pid, (sn, sa) in pairs.items():
                    out_r.append(q)
                    out_c.append(pid)
                    out_sn.append(sn)
                    out_sa.append(sa)

            if not out_r:
                continue
            r = np.array(out_r, dtype=np.int32)
            c = np.array(out_c, dtype=np.int32)
            sn = np.array(out_sn, dtype=np.float32)
            sa = np.array(out_sa, dtype=np.float32)
            print(f"[blocking] S1 {start}-{stop}: {len(r):,} pairs ({time.time()-t0:.1f}s)", flush=True)
            yield r, c, sn, sa


def blocking_stats(s1_idx, cand_ids, n_s1, gt_list=None, s1_ids=None, n_pool=None):
    """Blocking recall / candidate distribution. gt_list: list aligned with s1 rows."""
    cnt = np.bincount(s1_idx, minlength=n_s1)
    out = {"n_s1": int(n_s1), "total_pairs": int(len(s1_idx)),
           "mean_cands": float(cnt.mean()),
           "median_cands": float(np.median(cnt)),
           "p90_cands": float(np.percentile(cnt, 90)),
           "p95_cands": float(np.percentile(cnt, 95)),
           "p99_cands": float(np.percentile(cnt, 99)),
           "max_cands": int(cnt.max()) if n_s1 else 0,
           "zero_cand_s1_frac": float((cnt == 0).mean())}
    if n_pool:
        out["reduction_ratio"] = 1.0 - len(s1_idx) / (n_s1 * n_pool)
    if gt_list is not None:
        cand_set = set(zip(s1_idx.tolist(), cand_ids))
        tot = hit = 0
        per = []
        for k, g in enumerate(gt_list):
            if g:
                h = sum((k, x) in cand_set for x in g)
                tot += len(g)
                hit += h
                per.append(h / len(g))
        out["pair_recall"] = hit / tot if tot else 1.0
        out["macro_entity_recall"] = float(np.mean(per)) if per else 1.0
        out["entities_full_recall"] = float(np.mean([p == 1.0 for p in per])) if per else 1.0
    return out
