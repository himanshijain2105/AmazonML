"""Validation error dump + taxonomy + F0.5-loss attribution.

Every error row is tagged with (possibly several) categories. Each validation S1 loses
(1 - F0.5) of the metric; that loss is split equally across its error rows and, within a
row, equally across the row's categories. The summary therefore answers:
"which error categories cost the most official macro F0.5?"
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .normalize import LEGAL_TOKENS

_WEB = re.compile(r"\.(com|in|net|org|co)\b|www\.|^#")
_LEET = re.compile(r"[a-z][01][a-z]|^[01][a-z]{2}")


def _nonlatin(s):
    return any(ch.isalpha() and not ch.isascii() for ch in s) and not any(ch.isascii() and ch.isalpha() for ch in s)


def categorize(kind, s1_name, c_name, s1_core, c_core, s1_addr, c_addr, s1_nums, c_nums, n_true, name_freq):
    cats = []
    if kind == "FN_blocking_miss":
        cats.append("blocking_miss")
    elif kind == "FP_singleton":
        cats.append("singleton_false_positive")
    elif kind == "FP":
        cats.append("threshold_false_positive")
    else:
        cats.append("threshold_false_negative")
    if kind.startswith("FN") and n_true >= 2:
        cats.append("multi_match_miss")
    nr = fuzz.ratio(s1_core, c_core)
    ar = fuzz.token_set_ratio(s1_addr, c_addr) if s1_addr and c_addr else 0
    if kind.startswith("FP"):
        if nr >= 90:
            cats.append("same_name_different_business")
        if ar >= 90 and nr < 70:
            cats.append("same_address_different_business")
        if name_freq >= np.log1p(3):
            cats.append("common_or_chain_name")
    if _nonlatin(c_name):
        cats.append("cross_script_name")
    if _WEB.search(c_name.lower()) or (len(c_name.split()) == 1 and len(c_name) >= 12):
        cats.append("web_or_hashtag_name")
    if _LEET.search(c_name.lower()):
        cats.append("digit_letter_typo")
    lc = [t for t in c_core.split()[:-1] if t in LEGAL_TOKENS]
    if lc:
        cats.append("legal_token_position")
    if not c_addr:
        cats.append("missing_address")
    n1, n2 = set(s1_nums.split()), set(c_nums.split())
    if n1 and n2 and s1_nums.split()[0] != c_nums.split()[0]:
        cats.append("address_number_conflict")
    if 70 <= nr < 100 and s1_core != c_core:
        cats.append("name_typo_or_variant")
    if nr < 50 and not _nonlatin(c_name):
        cats.append("different_name_dba_like")
    return cats


def dump_and_summarize(path, s1, pool, ev_rows, g, cand, prob, keep, y, gtl, Xv, names, pe, log):
    fi = {n: k for k, n in enumerate(names)}
    S = lambda c, k: s1[c].values[k]
    Pc = lambda c, k: pool[c].values[k]
    nums = "addr_nums" if "addr_nums" in pool else "addr_canon"
    addr = "addr_canon2" if "addr_canon2" in pool else "addr_canon"
    rows = []
    for m in np.flatnonzero(keep != y):
        k = ev_rows[g[m]]
        kind = ("FP_singleton" if len(gtl[k]) == 0 else "FP") if keep[m] else "FN_scored"
        rows.append((g[m], kind, k, cand[m], float(prob[m]), float(Xv[m, fi["s1_core_freq_s1"]])))
    cset = set(zip(g.tolist(), cand.tolist()))
    for gi, k in enumerate(ev_rows):
        for c in gtl[k]:
            if (gi, int(c)) not in cset:
                rows.append((gi, "FN_blocking_miss", k, int(c), np.nan, np.nan))
    out = []
    for gi, kind, k, c, p, fr in rows:
        cats = categorize(kind, S("business_name", k), Pc("business_name", c), S("name_core", k), Pc("name_core", c),
                          S(addr, k), Pc(addr, c), S(nums, k), Pc(nums, c), len(gtl[k]), 0 if np.isnan(fr) else fr)
        out.append(dict(group=gi, kind=kind, s1=S("entity_id", k), cand=Pc("entity_id", c), prob=p, n_true=len(gtl[k]),
                        country=S("country", k), s1_name=S("business_name", k), cand_name=Pc("business_name", c),
                        s1_addr=S("business_address", k), cand_addr=Pc("business_address", c),
                        categories="|".join(cats)))
    df = pd.DataFrame(out)
    if df.empty:
        log("no validation errors")
        return
    loss = 1 - pe
    per_row = df.groupby("group")["group"].transform("size")
    df["f05_loss_share"] = loss[df["group"].values] / per_row.values
    df.drop(columns="group").to_csv(path, sep="\t", index=False)
    ex = df.assign(cat=df["categories"].str.split("|"))
    ex["share"] = ex["f05_loss_share"] / ex["cat"].map(len)
    ex = ex.explode("cat")
    summ = ex.groupby("cat").agg(rows=("cat", "size"), macro_f05_lost=("share", "sum"))
    summ["macro_f05_lost"] /= len(ev_rows)
    summ = summ.sort_values("macro_f05_lost", ascending=False)
    summ.to_csv(str(path).replace(".tsv", "_summary.tsv"), sep="\t")
    log(f"error taxonomy ({len(df):,} error rows; F0.5 lost = {loss.mean():.4f}):\n" + summ.to_string())
