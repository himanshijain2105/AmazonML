#!/usr/bin/env python3
"""Categorise blocking misses (or any error rows) and measure which retrieval key families the
S1 record and its true match actually share (a pair can only be retrieved via a shared key).

Usage:  python tools/analyze_misses.py work/misses_<EXP>.tsv
        python tools/analyze_misses.py work/val_errors_<EXP>.tsv --kind FN_blocking_miss
Input columns: s1_name, cand_name, s1_addr, cand_addr (+ optional kind).
"""
import argparse
import collections
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from ber.blocking import keys_v1, keys_v2  # noqa: E402
from ber.normalize import LEGAL_TOKENS, addr_forms, addr_forms_v2, is_nonlatin_name, name_forms, name_key  # noqa: E402


def fams(keys):
    return {k.split("|", 1)[1] for k, _ in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kind", default=None)
    a = ap.parse_args()
    e = pd.read_csv(a.path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    if a.kind and "kind" in e:
        e = e[e.kind == a.kind]
    N = len(e)
    if not N:
        print("no rows")
        return
    cats, share_v1, share_v2, spec_v1, spec_v2 = collections.Counter(), 0, 0, 0, 0
    fam_share = collections.Counter()
    for r in e.itertuples():
        _, c1 = name_forms(r.s1_name); _, c2 = name_forms(r.cand_name)
        a1, p1, h1 = addr_forms(r.s1_addr); a2, p2, h2 = addr_forms(r.cand_addr)
        k1 = fams(keys_v1("x", c1, a1, p1, h1)) & fams(keys_v1("x", c2, a2, p2, h2))
        v1 = addr_forms_v2(r.s1_addr); v2 = addr_forms_v2(r.cand_addr)
        q1 = keys_v2("x", name_key(r.s1_name), v1[0], v1[1], v1[3], v1[4])
        q2 = keys_v2("x", name_key(r.cand_name), v2[0], v2[1], v2[3], v2[4])
        k2 = fams(q1) & fams(q2)
        share_v1 += bool(k1); share_v2 += bool(k2)
        spec_v1 += any(x.split(":")[0] in ("c", "nz", "nh") for x in k1)
        s2 = {x.split(":")[0] for x in k2}
        spec_v2 += bool(s2 & {"c", "cp", "nz", "nn", "an", "al", "ci"})
        for f in s2:
            fam_share[f] += 1
        low = r.cand_name.lower()
        if is_nonlatin_name(r.cand_name) or any(ord(ch) > 127 and ch.isalpha() for ch in r.cand_name):
            cats["non_latin_cand_name"] += 1
        if re.search(r"\.com|www\.|^#|com$", low) or (len(r.cand_name.split()) == 1 and len(r.cand_name) >= 10):
            cats["web_or_hashtag_name"] += 1
        if re.search(r"[a-z][01][a-z]|\b[01][a-z]{2}", low):
            cats["digit_letter_typo"] += 1
        if not r.cand_addr.strip():
            cats["cand_addr_empty"] += 1
        if re.search(r"\b0\d+", r.cand_addr):
            cats["leading_zero_number"] += 1
        n1, n2 = set(v1[3].split()), set(v2[3].split())
        if n1 and n2 and not (n1 & n2):
            cats["no_shared_number"] += 1
        if any(t in LEGAL_TOKENS for t in c2.split()[:-1]):
            cats["legal_token_not_at_end"] += 1
        tsr = fuzz.token_set_ratio(c1, c2)
        if tsr < 50 and not is_nonlatin_name(r.cand_name):
            cats["different_name_dba_like"] += 1
        elif c1 != c2 and fuzz.ratio(c1, c2) >= 75:
            cats["name_typo_or_variant"] += 1
    print(f"rows analysed: {N}\n\nfailure categories (multi-label):")
    for k, v in cats.most_common():
        print(f"  {k:26s} {v:6d} ({v / N * 100:5.1f}%)")
    print(f"\nshare >=1 key  v1: {share_v1 / N * 100:5.1f}%   v2: {share_v2 / N * 100:5.1f}%")
    print(f"share a SPECIFIC key (compact/composite)  v1: {spec_v1 / N * 100:5.1f}%   v2: {spec_v2 / N * 100:5.1f}%")
    print("\nv2 key families shared by S1 and its true match:")
    for k, v in fam_share.most_common():
        print(f"  {k:4s} {v / N * 100:5.1f}%")


if __name__ == "__main__":
    main()
