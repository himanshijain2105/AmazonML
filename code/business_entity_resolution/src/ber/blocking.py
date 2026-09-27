"""Candidate generation: compact CSR inverted index, two scoring channels, versioned key families.

Memory design (8 GB Mac):
  offsets[2^hash_bits + 1] int64 | idf[2^hash_bits] float32
  doc_ids[P] int32 | fam[P] uint8       (P = kept postings; 5 bytes each)
  norm_name[N], norm_addr[N] float32
Posting weights are NOT stored: weight = FAMILY_WEIGHT[channel][fam] * idf[bucket] at query time.
Pass 1 hashes every record's keys once and spills them to disk-backed memmaps; pass 2 is a
vectorised counting-sort fill, chunk by chunk, so peak RAM is bounded.

Key versions:
  v1 = exact reproduction of the EXP-001 baseline keys (uses v1 normalized columns).
  v2 = families motivated by the real EXP-001 miss analysis (see normalize.py v2 notes):
       n  name-key tokens (de-leeted, legal-free anywhere, web-stripped)
       p  4-char prefix of name-key tokens (len>=5)            c  sorted compact name key
       cp 8-char prefix of in-order compact name key (domains/hashtags: "melurengineering.com")
       a  canonical address tokens (zero-stripped numbers)      z  postal code
       nz first name token x postal                             nn first name token x digit run (first 3)
       an digit run x next 2 address words (street name)       al digit run x last 2 address words (city/state)
       ci compound address id ("6/79", "8-2-616/1")
All keys are prefixed with the open-set country string (no hard-coded countries).
"""
from __future__ import annotations

import math
import os
import tempfile
import time
import zlib
from typing import Iterator, Tuple

import numpy as np

FAMILIES = ["n", "p", "c", "a", "z", "nz", "nh", "cp", "nn", "an", "al", "ci"]
FAM_ID = {f: k for k, f in enumerate(FAMILIES)}
PROFILES = {
    "name": {"n": 1.0, "p": 0.35, "c": 1.2, "a": 0.15, "z": 0.2, "nz": 0.8, "nh": 0.6,
             "cp": 0.9, "nn": 0.6, "an": 0.25, "al": 0.25, "ci": 0.3},
    "addr": {"n": 0.3, "p": 0.15, "c": 0.3, "a": 1.0, "z": 1.0, "nz": 1.0, "nh": 1.0,
             "cp": 0.3, "nn": 1.0, "an": 1.3, "al": 1.3, "ci": 1.3},
}
V2_FAMILIES = ("n", "p", "c", "cp", "a", "z", "nz", "nn", "an", "al", "ci")
_GENERIC_ADDR = frozenset("st rd ave blvd dr ln ct pl no fl unit ste apt plot nr near house h hno door "
                          "flat block bldg sec col ngr floor opp".split())


def keys_v1(country, core, addr, postal, house):
    c = country + "|"
    toks = core.split()
    out = []
    for t in toks:
        out.append((c + "n:" + t, 0))
        if len(t) >= 5:
            out.append((c + "p:" + t[:4], 1))
    if toks:
        out.append((c + "c:" + "".join(sorted(toks)), 2))
        if postal:
            out.append((c + "nz:" + toks[0] + "|" + postal, 5))
        if house:
            out.append((c + "nh:" + toks[0] + "|" + house, 6))
    for t in set(addr.split()):
        out.append((c + "a:" + t, 3))
    if postal:
        out.append((c + "z:" + postal, 4))
    return out


def keys_v2(country, nkey, addr, postal, nums, cids, families=V2_FAMILIES):
    c = country + "|"
    F = families
    toks = nkey.split()
    out = []
    if toks:
        if "n" in F:
            out += [(c + "n:" + t, 0) for t in toks]
        if "p" in F:
            out += [(c + "p:" + t[:4], 1) for t in toks if len(t) >= 5]
        if "c" in F:
            out.append((c + "c:" + "".join(sorted(toks)), 2))
        comp = "".join(toks)
        if "cp" in F and len(comp) >= 8:
            out.append((c + "cp:" + comp[:8], 7))
        if "nz" in F and postal:
            out.append((c + "nz:" + toks[0] + "|" + postal, 5))
    numl = [x for x in nums.split() if len(x) <= 5] if nums else []
    if toks and "nn" in F:
        out += [(c + "nn:" + toks[0] + "|" + x, 8) for x in dict.fromkeys(numl[:3])]
    at = addr.split()
    if "a" in F:
        out += [(c + "a:" + t, 3) for t in set(at)]
    if "z" in F and postal:
        out.append((c + "z:" + postal, 4))
    if ("an" in F or "al" in F) and at:
        words = [t for t in at if t.isalpha() and len(t) >= 3 and t not in _GENERIC_ADDR]
        seen_n = 0
        for k, t in enumerate(at):
            if t.isdigit() and len(t) <= 5:
                if "an" in F:
                    nxt = [y for y in at[k + 1:k + 6] if y.isalpha() and len(y) >= 3 and y not in _GENERIC_ADDR][:2]
                    out += [(c + "an:" + t + "|" + y, 9) for y in nxt]
                if "al" in F and seen_n < 2:
                    out += [(c + "al:" + t + "|" + y, 10) for y in words[-2:]]
                seen_n += 1
    if "ci" in F and cids:
        out += [(c + "ci:" + x, 11) for x in cids.split()]
    return out


def make_key_fn(version, families=None):
    """Return f(df) -> iterator over per-record key lists, reading the right normalized columns."""
    if version == 1:
        def gen(df):
            return (keys_v1(*r) for r in zip(df["country_n"].values, df["name_core"].values,
                                               df["addr_canon"].values, df["postal"].values,
                                               df["house"].values))
        return gen
    fam = tuple(families) if families else V2_FAMILIES

    def gen(df):
        return (keys_v2(*r, families=fam) for r in zip(df["country_n"].values, df["name_key"].values,
                                                         df["addr_canon2"].values, df["postal2"].values,
                                                         df["addr_nums"].values, df["addr_cids"].values))
    return gen


def _h(key, nf_mask):
    return zlib.crc32(key.encode("utf-8")) & nf_mask


class CompactIndex:
    def __init__(self, hash_bits=24, max_df=5000, spill_dir=None):
        self.hash_bits, self.nf = hash_bits, 1 << hash_bits
        self.max_df = max_df
        self.spill_dir = spill_dir
        self.W = {ch: np.array([PROFILES[ch][f] for f in FAMILIES], np.float32) for ch in PROFILES}

    def build(self, pool_df, key_fn, log=print):
        t0 = time.time()
        n = len(pool_df)
        self.n_docs = n
        mask = self.nf - 1
        tmp = tempfile.mkdtemp(dir=self.spill_dir)
        cap = max(1 << 20, n * 32)
        kfile, ffile = os.path.join(tmp, "k.u32"), os.path.join(tmp, "f.u8")
        K = np.memmap(kfile, dtype=np.uint32, mode="w+", shape=(cap,))
        Fm = np.memmap(ffile, dtype=np.uint8, mode="w+", shape=(cap,))
        cnt = np.zeros(n, np.int32)
        df = np.zeros(self.nf, np.int64)
        pos = 0
        buf_k, buf_f = [], []
        report = max(1, n // 10)
        # ---- pass 1: hash keys once, dedupe per record, spill to disk, count df
        for ri, keys in enumerate(key_fn(pool_df)):
            seen = set()
            m = 0
            for ks, fam in keys:
                hv = _h(ks, mask)
                if hv not in seen:
                    seen.add(hv)
                    buf_k.append(hv)
                    buf_f.append(fam)
                    m += 1
            cnt[ri] = m
            if len(buf_k) >= 4_000_000 or ri == n - 1:
                a = np.fromiter(buf_k, np.uint32, len(buf_k))
                if pos + len(a) > cap:  # grow spill files
                    cap = int((pos + len(a)) * 1.5)
                    K.flush(); Fm.flush(); del K, Fm
                    K = np.memmap(kfile, dtype=np.uint32, mode="r+", shape=(cap,))
                    Fm = np.memmap(ffile, dtype=np.uint8, mode="r+", shape=(cap,))
                K[pos:pos + len(a)] = a
                Fm[pos:pos + len(a)] = np.fromiter(buf_f, np.uint8, len(buf_f))
                df += np.bincount(a, minlength=self.nf)
                pos += len(a)
                buf_k, buf_f = [], []
            if (ri + 1) % report == 0:
                log(f"  [index pass1] {ri + 1:,}/{n:,} keys={pos:,} ({time.time() - t0:.0f}s)")
        total_keys = pos
        idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
        keep_bucket = (df > 0) & (df <= self.max_df)
        idf[~keep_bucket] = 0.0
        self.idf = idf
        kept_df = np.where(keep_bucket, df, 0)
        self.offsets = np.zeros(self.nf + 1, np.int64)
        np.cumsum(kept_df, out=self.offsets[1:])
        P = int(self.offsets[-1])
        self.stats = {"pool": n, "hash_bits": self.hash_bits, "keys_emitted": int(total_keys),
                      "keys_per_record": total_keys / max(n, 1), "buckets_used": int((df > 0).sum()),
                      "buckets_dropped_max_df": int((df > self.max_df).sum()),
                      "postings_dropped_max_df": int(df[df > self.max_df].sum()), "postings_kept": P,
                      "index_mb": (P * 5 + self.nf * 12 + n * 8) / 2 ** 20}
        del df, kept_df
        # ---- pass 2: vectorised fill (counting sort per chunk)
        self.doc_ids = np.empty(P, np.int32)
        self.fam = np.empty(P, np.uint8)
        cursor = self.offsets[:-1].copy()
        n2n = np.zeros(n, np.float64)
        n2a = np.zeros(n, np.float64)
        rec_start = np.zeros(n + 1, np.int64)
        np.cumsum(cnt, out=rec_start[1:])
        step = 1_000_000
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            a, b = rec_start[r0], rec_start[r1]
            kk = np.asarray(K[a:b]).astype(np.int64)
            ff = np.asarray(Fm[a:b])
            dd = np.repeat(np.arange(r0, r1, dtype=np.int32), cnt[r0:r1])
            ok = idf[kk] > 0
            kk, ff, dd = kk[ok], ff[ok], dd[ok]
            for ch, acc in (("name", n2n), ("addr", n2a)):
                w = self.W[ch][ff] * idf[kk]
                acc[r0:r1] += np.bincount(dd - r0, weights=w.astype(np.float64) ** 2, minlength=r1 - r0)
            o = np.argsort(kk, kind="stable")
            ks = kk[o]
            starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
            lens = np.diff(np.r_[starts, len(ks)])
            rank = np.arange(len(ks)) - np.repeat(starts, lens)
            dst = cursor[ks] + rank
            self.doc_ids[dst] = dd[o]
            self.fam[dst] = ff[o]
            cursor[ks[starts]] += lens
            log(f"  [index pass2] {r1:,}/{n:,} ({time.time() - t0:.0f}s)")
        del K, Fm
        for f in (kfile, ffile):
            os.remove(f)
        os.rmdir(tmp)
        self.norm = {"name": np.sqrt(n2n).astype(np.float32), "addr": np.sqrt(n2a).astype(np.float32)}
        for v in self.norm.values():
            v[v == 0] = 1.0
        self.stats["build_seconds"] = time.time() - t0
        log(f"[index] {self.stats}")
        return self

    def query_one(self, keys, k_name, k_addr, min_score):
        """-> (pool_ids, cos_name, cos_addr, found_by) for the union of both channel top-ks."""
        mask = self.nf - 1
        q = {}
        for ks, fam in keys:
            hv = _h(ks, mask)
            if hv not in q and self.idf[hv] > 0:
                q[hv] = fam
        if not q:
            return None
        hs = np.fromiter(q.keys(), np.int64, len(q))
        fq = np.fromiter(q.values(), np.int64, len(q))
        idf_q = self.idf[hs]
        qw = {ch: self.W[ch][fq] * idf_q for ch in PROFILES}
        s, e = self.offsets[hs], self.offsets[hs + 1]
        lens = e - s
        if lens.sum() == 0:
            return None
        idx = np.concatenate([np.arange(a, b) for a, b in zip(s, e) if b > a])
        term = np.repeat(np.arange(len(hs)), lens)
        docs = self.doc_ids[idx]
        pf = self.fam[idx]
        u, inv = np.unique(docs, return_inverse=True)
        cos = {}
        for ch in PROFILES:
            contrib = (qw[ch][term] * self.W[ch][pf] * idf_q[term]).astype(np.float64)
            raw = np.bincount(inv, weights=contrib, minlength=len(u))
            qn = math.sqrt(float((qw[ch] ** 2).sum())) or 1.0
            cos[ch] = (raw / (qn * self.norm[ch][u])).astype(np.float32)
        sel = np.zeros(len(u), np.uint8)
        for ch, k, bit in (("name", k_name, 1), ("addr", k_addr, 2)):
            c = cos[ch]
            ok = np.flatnonzero(c >= min_score)
            if len(ok) > k:
                ok = ok[np.argpartition(-c[ok], k - 1)[:k]]
            sel[ok] |= bit
        keep = np.flatnonzero(sel)
        return u[keep].astype(np.int32), cos["name"][keep], cos["addr"][keep], sel[keep]


class Blocker:
    def __init__(self, hash_bits=24, max_df=5000, top_k=40, min_score=0.05, addr_share=0.4,
                 key_version=2, families=None, spill_dir=None, **_):
        self.hash_bits, self.max_df, self.min_score = hash_bits, max_df, min_score
        self.top_k, self.key_version, self.families = top_k, key_version, families
        self.spill_dir = spill_dir
        self.set_top_k(top_k, addr_share)
        self.key_fn = make_key_fn(key_version, families)

    def set_top_k(self, top_k, addr_share=0.4):
        """top_k can be changed WITHOUT rebuilding the index (used by the sweep)."""
        self.top_k = top_k
        self.k_addr = max(1, int(round(top_k * addr_share)))
        self.k_name = max(1, top_k - self.k_addr)

    def fit_pool(self, pool_df, log=print):
        self.n_pool = len(pool_df)
        self.index = CompactIndex(self.hash_bits, self.max_df, self.spill_dir).build(pool_df, self.key_fn, log)
        return self

    def query(self, s1_df, chunk=5000) -> Iterator[Tuple[np.ndarray, ...]]:
        """Yield (s1_row, pool_row, cos_name, cos_addr, found_by) per chunk of S1 rows.
        found_by: 1 = name channel top-k, 2 = address channel top-k, 3 = both."""
        n = len(s1_df)
        for start in range(0, n, chunk):
            sub = s1_df.iloc[start:start + chunk]
            R, C, SN, SA, FB = [], [], [], [], []
            for q, keys in enumerate(self.key_fn(sub)):
                res = self.index.query_one(keys, self.k_name, self.k_addr, self.min_score)
                if res is None:
                    continue
                ids, cn, ca, fb = res
                R.append(np.full(len(ids), start + q, np.int32)); C.append(ids)
                SN.append(cn); SA.append(ca); FB.append(fb)
            if R:
                yield (np.concatenate(R), np.concatenate(C), np.concatenate(SN),
                       np.concatenate(SA), np.concatenate(FB))


def blocking_stats(s1_idx, cand_pool_idx, n_s1, gt_pool_idx, found_by=None, n_pool=None):
    """gt_pool_idx: list (aligned with S1 rows) of arrays of pool row indices of true matches.
    Pure-numpy recall computation (no Python set of tuples)."""
    cnt = np.bincount(s1_idx, minlength=n_s1)
    out = {"n_s1": int(n_s1), "total_pairs": int(len(s1_idx)), "mean_cands": float(cnt.mean()),
           "median_cands": float(np.median(cnt)), "p90_cands": float(np.percentile(cnt, 90)),
           "p95_cands": float(np.percentile(cnt, 95)), "p99_cands": float(np.percentile(cnt, 99)),
           "max_cands": int(cnt.max()) if n_s1 else 0, "zero_cand_s1_frac": float((cnt == 0).mean())}
    if n_pool:
        out["reduction_ratio"] = 1.0 - len(s1_idx) / (n_s1 * n_pool)
    n_true = np.array([len(g) for g in gt_pool_idx])
    gi = np.repeat(np.arange(n_s1), n_true)
    gj = np.concatenate([np.asarray(g, np.int64) for g in gt_pool_idx]) if n_true.sum() else np.zeros(0, np.int64)
    key_c = s1_idx.astype(np.int64) * (1 << 32) + cand_pool_idx.astype(np.int64)
    key_g = gi.astype(np.int64) * (1 << 32) + gj
    order = np.argsort(key_c)
    pos = np.searchsorted(key_c[order], key_g)
    pos = np.minimum(pos, max(len(key_c) - 1, 0))
    hit = (len(key_c) > 0) & (key_c[order][pos] == key_g) if len(key_c) else np.zeros(len(key_g), bool)
    out["pair_recall"] = float(hit.mean()) if len(hit) else 1.0
    per_hit = np.bincount(gi, weights=hit, minlength=n_s1)
    has = n_true > 0
    out["macro_entity_recall"] = float((per_hit[has] / n_true[has]).mean()) if has.any() else 1.0
    out["entities_full_recall"] = float((per_hit[has] == n_true[has]).mean()) if has.any() else 1.0
    if found_by is not None and len(key_c):
        fb = found_by[order][pos]
        out["pair_recall_name_channel"] = float((hit & ((fb & 1) > 0)).mean())
        out["pair_recall_addr_channel"] = float((hit & ((fb & 2) > 0)).mean())
        out["pair_recall_addr_only"] = float((hit & (fb == 2)).mean())
    return out, hit
