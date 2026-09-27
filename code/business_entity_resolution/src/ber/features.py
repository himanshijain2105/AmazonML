"""Vectorised pairwise features; ONE function for training, validation and inference.

Memory: no pool-wide sparse matrices are kept. Pool-level IDF is computed once by streaming
hashed tokens (2^22 float32 arrays); per chunk, small sparse matrices are built only for the
S1 rows and pool rows that actually occur in that chunk's candidate pairs.

Feature sets:
  v1 = the EXP-001 baseline features (34), reproducible.
  v2 = v1 + features motivated by the real EXP-001 error analysis:
       digit-run number agreement/conflict (near-duplicate hard negatives: "415 vs 417 Mccurdy Ave",
       "PLOTNO165 vs PLOTNO172", "6th Flr-602 vs 8th Flr-602"), cross-script name flags (22-24% of
       scored errors have non-Latin candidate names), name-key / compact-prefix equality
       (web/hashtag/leet forms), compound-id and number x street-word composite overlap,
       and which blocking channel retrieved the pair.
No feature encodes the country VALUE (open-set safe).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction import FeatureHasher

V1 = ["blk_name", "blk_addr", "blk_rank_name", "blk_rank_addr", "blk_ratio_to_best", "n_cands",
      "name_ratio", "name_tsort", "core_ratio", "core_tset", "core_partial", "core_jw", "core_exact",
      "name_idf_cos", "name_tok_jacc", "first_tok_eq", "name_len_ratio",
      "addr_ratio", "addr_tset", "addr_partial", "addr_idf_cos", "num_jacc",
      "postal_eq", "postal_conflict", "house_eq", "house_conflict", "cand_addr_empty",
      "is_s3", "s1_core_freq_s1", "s1_core_freq_pool",
      "grp_n_core_exact", "grp_rank_name", "grp_rank_addr", "grp_best_addr_given_name"]
V2_EXTRA = ["found_by", "key_exact", "key_tset", "key_jw", "cp8_eq",
            "s1_nonlatin", "cand_nonlatin", "script_mismatch",
            "dig_jacc", "dig_first_eq", "dig_first_conflict", "dig_first_absdiff_small",
            "n_dig_only_s1", "n_dig_only_cand", "cid_shared", "cid_conflict", "comp_shared",
            "grp_n_key_exact", "grp_rank_prob_proxy"]
FEATURE_SETS = {"v1": V1, "v2": V1 + V2_EXTRA}
FEATURE_NAMES = FEATURE_SETS["v2"]          # default; the model json records the set actually used
_H = 2 ** 22
_HASHER = FeatureHasher(n_features=_H, input_type="string", alternate_sign=False, dtype=np.float32)


def _bin(token_lists):
    X = _HASHER.transform(token_lists).tocsr()
    X.data[:] = 1.0
    return X


def _idf_stream(texts, n, step=1_000_000):
    df = np.zeros(_H, np.int64)
    for a in range(0, n, step):
        X = _bin(list(set(s.split())) for s in texts[a:a + step])
        df += np.bincount(X.indices, minlength=_H)
    return (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)


def _weighted(X, idf):
    X = X.copy()
    X.data *= idf[X.indices]
    nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    nr[nr == 0] = 1
    return sp.diags(1 / nr).dot(X).tocsr()


def _composites(addr):
    """digit run x next 2 address words (same definition as blocking key family 'an')."""
    at = addr.split()
    out = []
    for k, t in enumerate(at):
        if t.isdigit() and len(t) <= 5:
            out += [t + "|" + y for y in [y for y in at[k + 1:k + 6] if y.isalpha() and len(y) >= 3][:2]]
    return out


class FeatureContext:
    def __init__(self, s1, pool, feature_set="v2"):
        self.fs = feature_set
        self.names = FEATURE_SETS[feature_set]
        self.v2 = feature_set == "v2"
        acol = "addr_canon2" if self.v2 else "addr_canon"
        self.acol = acol
        self.s1, self.pool = s1, pool
        n = len(pool)
        self.idf_n = _idf_stream(pool["name_core"].values, n)
        self.idf_a = _idf_stream(pool[acol].values, n)
        k1 = s1["country_n"].values + "|" + np.array([x.replace(" ", "") for x in s1["name_core"].values], dtype=object)
        vc1 = {}
        for k in k1:
            vc1[k] = vc1.get(k, 0) + 1
        wanted = set(k1)
        vcp = dict.fromkeys(wanted, 0)
        for c, x in zip(pool["country_n"].values, pool["name_core"].values):
            k = c + "|" + x.replace(" ", "")
            if k in vcp:
                vcp[k] += 1
        self.s1_freq_s1 = np.log1p(np.array([vc1[k] for k in k1], np.float32))
        self.s1_freq_pool = np.log1p(np.array([vcp[k] for k in k1], np.float32))
        self.is_s3 = pool["entity_id"].str.startswith("S3-").to_numpy(dtype=np.float32)

    # ---- per-chunk helpers -------------------------------------------------------------
    def _col(self, side, col, idx):
        df = self.s1 if side == "s1" else self.pool
        return df[col].values[idx]


def _rowdot(A, B):
    return np.asarray(A.multiply(B).sum(1)).ravel().astype(np.float32)


def _cp(a, b, scorer):
    return (cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / 100.0).astype(np.float32)


def _group_rank(g, v):
    order = np.lexsort((-v, g))
    rank = np.empty(len(v), np.float32)
    starts = np.r_[0, np.flatnonzero(np.diff(g[order])) + 1]
    pos = np.arange(len(v)) - np.repeat(starts, np.diff(np.r_[starts, len(v)]))
    rank[order] = pos
    return rank


def _set_features(a_str, b_str):
    """Binary token matrices for two aligned string arrays -> (inter, n_a, n_b)."""
    A = _bin(list(set(s.split())) for s in a_str)
    B = _bin(list(set(s.split())) for s in b_str)
    inter = _rowdot(A, B)
    na = np.asarray(A.sum(1)).ravel().astype(np.float32)
    nb = np.asarray(B.sum(1)).ravel().astype(np.float32)
    return inter, na, nb


def pair_features(ctx: FeatureContext, i, j, sn, sa, found_by=None) -> np.ndarray:
    """i must be grouped (sorted) by S1 row. Returns float32 [len(i), len(ctx.names)]."""
    F = {}
    m = len(i)
    F["blk_name"], F["blk_addr"] = sn, sa
    F["blk_rank_name"] = _group_rank(i, sn)
    F["blk_rank_addr"] = _group_rank(i, sa)
    blk = np.maximum(sn, sa)
    uniq, inv, cnt = np.unique(i, return_inverse=True, return_counts=True)
    best = np.zeros(len(uniq), np.float32)
    np.maximum.at(best, inv, blk)
    F["blk_ratio_to_best"] = blk / np.maximum(best[inv], 1e-6)
    F["n_cands"] = cnt[inv].astype(np.float32)

    S = lambda c: ctx._col("s1", c, i)
    P = lambda c: ctx._col("pool", c, j)
    a_nc, b_nc = S("name_clean"), P("name_clean")
    a_co, b_co = S("name_core"), P("name_core")
    F["name_ratio"] = _cp(a_nc, b_nc, fuzz.ratio)
    F["name_tsort"] = _cp(a_nc, b_nc, fuzz.token_sort_ratio)
    F["core_ratio"] = _cp(a_co, b_co, fuzz.ratio)
    F["core_tset"] = _cp(a_co, b_co, fuzz.token_set_ratio)
    F["core_partial"] = _cp(a_co, b_co, fuzz.partial_ratio)
    a_cp = np.array([x.replace(" ", "") for x in a_co], dtype=object)
    b_cp = np.array([x.replace(" ", "") for x in b_co], dtype=object)
    F["core_jw"] = cpdist(list(a_cp), list(b_cp), scorer=JaroWinkler.normalized_similarity,
                          workers=-1, dtype=np.float32).astype(np.float32)
    F["core_exact"] = ((a_cp == b_cp) & (a_cp != "")).astype(np.float32)
    An = _bin(list(set(s.split())) for s in a_co)
    Bn = _bin(list(set(s.split())) for s in b_co)
    F["name_idf_cos"] = _rowdot(_weighted(An, ctx.idf_n), _weighted(Bn, ctx.idf_n))
    inter = _rowdot(An, Bn)
    n1 = np.asarray(An.sum(1)).ravel()
    n2 = np.asarray(Bn.sum(1)).ravel()
    F["name_tok_jacc"] = inter / np.maximum(n1 + n2 - inter, 1)
    f1 = np.array([x.split(" ", 1)[0] for x in a_co], dtype=object)
    f2 = np.array([x.split(" ", 1)[0] for x in b_co], dtype=object)
    F["first_tok_eq"] = ((f1 == f2) & (f1 != "")).astype(np.float32)
    l1 = np.fromiter((len(x) for x in a_cp), np.float32, m)
    l2 = np.fromiter((len(x) for x in b_cp), np.float32, m)
    F["name_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)

    a_ad, b_ad = S(ctx.acol), P(ctx.acol)
    F["addr_ratio"] = _cp(a_ad, b_ad, fuzz.ratio)
    F["addr_tset"] = _cp(a_ad, b_ad, fuzz.token_set_ratio)
    F["addr_partial"] = _cp(a_ad, b_ad, fuzz.partial_ratio)
    Aa = _bin(list(set(s.split())) for s in a_ad)
    Ba = _bin(list(set(s.split())) for s in b_ad)
    F["addr_idf_cos"] = _rowdot(_weighted(Aa, ctx.idf_a), _weighted(Ba, ctx.idf_a))
    num_a = [" ".join(t for t in s.split() if t.isdigit()) for s in a_ad]
    num_b = [" ".join(t for t in s.split() if t.isdigit()) for s in b_ad]
    ni, na, nb = _set_features(num_a, num_b)
    F["num_jacc"] = np.where(na + nb - ni > 0, ni / np.maximum(na + nb - ni, 1), -1).astype(np.float32)
    pcol, hcol = ("postal2", "house2") if ctx.v2 else ("postal", "house")   # v2: zero-padding-safe columns
    p1, p2 = S(pcol), P(pcol)
    both = (p1 != "") & (p2 != "")
    F["postal_eq"] = (both & (p1 == p2)).astype(np.float32)
    F["postal_conflict"] = (both & (p1 != p2)).astype(np.float32)
    h1, h2 = S(hcol), P(hcol)
    hb = (h1 != "") & (h2 != "")
    F["house_eq"] = (hb & (h1 == h2)).astype(np.float32)
    F["house_conflict"] = (hb & (h1 != h2)).astype(np.float32)
    F["cand_addr_empty"] = (b_ad == "").astype(np.float32)
    F["is_s3"] = ctx.is_s3[j]
    F["s1_core_freq_s1"] = ctx.s1_freq_s1[i]
    F["s1_core_freq_pool"] = ctx.s1_freq_pool[i]
    ce = F["core_exact"]
    g_ce = np.zeros(len(uniq), np.float32)
    np.add.at(g_ce, inv, ce)
    F["grp_n_core_exact"] = g_ce[inv]
    F["grp_rank_name"] = _group_rank(i, F["core_ratio"])
    F["grp_rank_addr"] = _group_rank(i, F["addr_ratio"])
    strong = np.where(F["core_ratio"] >= 0.85, F["addr_ratio"], -1).astype(np.float32)
    gb = np.full(len(uniq), -1, np.float32)
    np.maximum.at(gb, inv, strong)
    F["grp_best_addr_given_name"] = F["addr_ratio"] - gb[inv]

    if ctx.v2:
        F["found_by"] = found_by.astype(np.float32) if found_by is not None else np.zeros(m, np.float32)
        a_k, b_k = S("name_key"), P("name_key")
        F["key_exact"] = ((a_k == b_k) & (a_k != "")).astype(np.float32)
        F["key_tset"] = _cp(a_k, b_k, fuzz.token_set_ratio)
        a_kc = [x.replace(" ", "") for x in a_k]
        b_kc = [x.replace(" ", "") for x in b_k]
        F["key_jw"] = cpdist(a_kc, b_kc, scorer=JaroWinkler.normalized_similarity, workers=-1,
                             dtype=np.float32).astype(np.float32)
        c8a = np.array([x[:8] if len(x) >= 8 else "" for x in a_kc], dtype=object)
        c8b = np.array([x[:8] if len(x) >= 8 else "" for x in b_kc], dtype=object)
        F["cp8_eq"] = ((c8a == c8b) & (c8a != "")).astype(np.float32)
        nl1, nl2 = S("name_nonlatin").astype(np.float32), P("name_nonlatin").astype(np.float32)
        F["s1_nonlatin"], F["cand_nonlatin"] = nl1, nl2
        F["script_mismatch"] = (nl1 != nl2).astype(np.float32)
        d1, d2 = S("addr_nums"), P("addr_nums")
        di, dn1, dn2 = _set_features(d1, d2)
        F["dig_jacc"] = np.where(dn1 + dn2 - di > 0, di / np.maximum(dn1 + dn2 - di, 1), -1).astype(np.float32)
        fd1 = np.array([x.split(" ", 1)[0] for x in d1], dtype=object)
        fd2 = np.array([x.split(" ", 1)[0] for x in d2], dtype=object)
        bothd = (fd1 != "") & (fd2 != "")
        F["dig_first_eq"] = (bothd & (fd1 == fd2)).astype(np.float32)
        F["dig_first_conflict"] = (bothd & (fd1 != fd2)).astype(np.float32)
        v1n = np.array([int(x) if x and len(x) < 10 else -10 ** 9 for x in fd1], np.int64)
        v2n = np.array([int(x) if x and len(x) < 10 else -10 ** 9 for x in fd2], np.int64)
        F["dig_first_absdiff_small"] = (bothd & (fd1 != fd2) & (np.abs(v1n - v2n) <= 20)).astype(np.float32)
        F["n_dig_only_s1"] = (dn1 - di).astype(np.float32)
        F["n_dig_only_cand"] = (dn2 - di).astype(np.float32)
        c1, c2 = S("addr_cids"), P("addr_cids")
        ci, cn1, cn2 = _set_features(c1, c2)
        F["cid_shared"] = (ci > 0).astype(np.float32)
        F["cid_conflict"] = ((cn1 > 0) & (cn2 > 0) & (ci == 0)).astype(np.float32)
        comp_i, _, _ = _set_features([" ".join(_composites(s)) for s in a_ad], [" ".join(_composites(s)) for s in b_ad])
        F["comp_shared"] = comp_i
        g_ke = np.zeros(len(uniq), np.float32)
        np.add.at(g_ke, inv, F["key_exact"])
        F["grp_n_key_exact"] = g_ke[inv]
        proxy = 0.5 * F["key_tset"] + 0.5 * F["addr_tset"]
        F["grp_rank_prob_proxy"] = _group_rank(i, proxy.astype(np.float32))
    return np.column_stack([F[k] for k in ctx.names]).astype(np.float32)
