"""Text normalization. All representations are ADDED; raw text is never overwritten.

Design rules (fixing defects found in the audit):
- Legal suffixes are stripped ONLY from the END of the name, iteratively, and never down to
  an empty string ("Private Eye Investigations LLC" -> "private eye investigations",
  not "eye investigations").
- Dotted acronyms are collapsed before punctuation removal ("L.L.C." -> "llc", "S.A.S." -> "sas").
- Address abbreviations are canonicalised to ONE form in BOTH directions ("Street"/"St" -> "st"),
  and the canonical form is what features and blocking actually use.
- Country is an open-set string (never hard-coded / one-hot).
"""
from __future__ import annotations

import re

import numpy as np
import unicodedata
from typing import List, Tuple

_RE_ACRONYM = re.compile(r"^(?:[a-z]\.){2,}[a-z]?\.?$")
_RE_NONALNUM = re.compile(r"[^a-z0-9]+")
_RE_DIGITS = re.compile(r"^\d+$")

LEGAL_TOKENS = frozenset("""
inc incorporated llc llp lp ltd limited corp corporation co company pvt private plc pllc pc
opc sarl sas sasu sa eurl sci snc gmbh ag bv nv srl spa pty pte
""".split())
# multi-token legal phrases collapsed to a single token before suffix stripping
_LEGAL_PHRASES = [("private limited", "pvtltd"), ("pvt ltd", "pvtltd"), ("pvt limited", "pvtltd"),
                  ("private ltd", "pvtltd"), ("limited liability company", "llc"),
                  ("limited liability partnership", "llp")]
LEGAL_TOKENS = LEGAL_TOKENS | {"pvtltd"}
NAME_STOP = frozenset(["the", "and", "of", "m", "s"])  # "M/s" (Messrs) is a common Indian prefix

ADDR_CANON = {
    # thoroughfares -> short canonical form (both directions map to the same token)
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "bvd": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
    "highway": "hwy", "parkway": "pkwy", "square": "sq", "circle": "cir", "terrace": "ter",
    "suite": "ste", "apartment": "apt", "floor": "fl", "building": "bldg", "room": "rm",
    "north": "n", "south": "s", "east": "e", "west": "w",
    # India
    "sector": "sec", "sect": "sec", "colony": "col", "clny": "col", "nagar": "ngr", "market": "mkt",
    "mrkt": "mkt", "station": "stn", "junction": "jn", "near": "nr", "opposite": "opp",
    "number": "no", "plot": "plot", "extension": "extn", "ext": "extn", "marg": "marg",
    # France
    "r": "rue", "allee": "all", "chemin": "ch", "impasse": "imp", "batiment": "bat",
    "residence": "res", "route": "rte",
}
ADDR_STOP = frozenset(["de", "la", "le", "les", "du", "des", "d", "l", "the", "of", "and"])


def fold(s: str) -> str:
    """lowercase, strip accents, collapse acronyms, '&'->'and', punctuation->space."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    toks = []
    for t in s.split():
        if "." in t and _RE_ACRONYM.match(t):
            t = t.replace(".", "")
        toks.append(t)
    return " ".join(_RE_NONALNUM.sub(" ", " ".join(toks)).split())


def name_forms(raw: str) -> Tuple[str, str]:
    """-> (name_clean, name_core). core = clean minus trailing legal suffix tokens and stopwords."""
    clean = fold(raw)
    if not clean:
        return "", ""
    s = " " + clean + " "
    for a, b in _LEGAL_PHRASES:
        if a in s:
            s = s.replace(" " + a + " ", " " + b + " ")
    toks = s.split()
    for sep in ("dba",):  # keep the primary name before a DBA clause
        if sep in toks[1:-1]:
            toks = toks[:toks.index(sep)]
    while len(toks) > 1 and (toks[-1] in LEGAL_TOKENS or toks[-1] == "and"):
        toks.pop()
    core_toks = [t for t in toks if t not in NAME_STOP] or toks
    return clean, " ".join(core_toks)


def addr_forms(raw: str) -> Tuple[str, str, str]:
    """-> (addr_canon, postal, house_no). postal = last 5-6 digit token; house = first 1-5 digit token."""
    clean = fold(raw)
    if not clean:
        return "", "", ""
    out: List[str] = []
    postal = house = ""
    for t in clean.split():
        t = ADDR_CANON.get(t, t)
        if t in ADDR_STOP:
            continue
        out.append(t)
        if _RE_DIGITS.match(t):
            if 5 <= len(t) <= 6:
                postal = t
            elif not house and len(t) <= 5:
                house = t
    # PIN codes written as "302 020"
    if not postal:
        m = re.search(r"\b(\d{3}) (\d{3})\b", " ".join(out))
        if m:
            postal = m.group(1) + m.group(2)
    return " ".join(out), postal, house


def normalize_frame(df):
    """Add normalized columns to a source DataFrame (in place) and return it."""
    nf = [name_forms(x) for x in df["business_name"].values]
    af = [addr_forms(x) for x in df["business_address"].values]
    df["name_clean"] = [a for a, _ in nf]
    df["name_core"] = [b for _, b in nf]
    df["addr_canon"] = [a for a, _, _ in af]
    df["postal"] = [b for _, b, _ in af]
    df["house"] = [c for _, _, c in af]
    df["country_n"] = [" ".join(fold(c).split()) for c in df["country"].values]
    return df


# =====================================================================================
# v2 representations (added columns; v1 columns are kept so the baseline stays reproducible)
# Each rule below is motivated by the REAL EXP-001 blocking-miss analysis (861 missed pairs):
#   leading-zero / zero-padded numbers ("002121 Addison" -> house 2121, not postal)  12.9%
#   0/1 digit-for-letter substitutions in names ("Simps0n", "A1egra", "We1fare")       5.8%
#   web / hashtag forms ("melurengineering.com", "#westcollege")                      22.5%
#   legal tokens not at the end ("LLC SILVER NET", "A1egra Limited Private Lal")      14.6%
#   compound address identifiers ("6/79", "8-2-616/1", "BJ-99")                         9.9%
# =====================================================================================
_RE_DIGRUN = re.compile(r"\d+")
_RE_LEET_MID = re.compile(r"(?<=[a-z])[01](?=[a-z])")
_RE_LEET_START = re.compile(r"^[01](?=[a-z]{2})")
_RE_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")
_RE_CID = re.compile(r"(?<![\w])([a-z0-9]*\d[a-z0-9]*(?:\s*[-/]\s*[a-z0-9]+)+)(?![\w])")
WEB_TOKENS = frozenset(["www", "http", "https"])
TLD_TOKENS = frozenset(["com", "in", "net", "org", "co", "biz", "info", "io"])
_RE_DOMAIN = re.compile(r"\.(com|in|net|org|co|biz|info|io)\b|^www\.|\bwww\.")
_LEET = str.maketrans({"0": "o", "1": "l"})


def deleet(tok: str) -> str:
    """0->o, 1->l only where a digit sits BETWEEN letters, or starts a token followed by
    two letters ("1ease"). Ordinals ("1st") and ordinary alphanumerics ("b4", "3m") untouched."""
    if not tok or tok.isdigit() or tok.isalpha() or _RE_ORDINAL.match(tok):
        return tok
    tok = _RE_LEET_MID.sub(lambda m: m.group(0).translate(_LEET), tok)
    return _RE_LEET_START.sub(lambda m: m.group(0).translate(_LEET), tok)


def name_key(raw: str) -> str:
    """Retrieval-oriented name form: de-leeted tokens, web prefixes/TLD suffixes removed,
    legal tokens and stopwords removed ANYWHERE (retrieval only — the model still sees name_core)."""
    toks = [deleet(t) for t in fold(raw).split()]
    while toks and toks[0] in WEB_TOKENS:
        toks = toks[1:]
    if _RE_DOMAIN.search(raw.lower()):   # only strip a TLD when the raw name is written as a domain
        while len(toks) > 1 and toks[-1] in TLD_TOKENS:
            toks = toks[:-1]
    s = " " + " ".join(toks) + " "
    for a, b in _LEGAL_PHRASES:
        s = s.replace(" " + a + " ", " " + b + " ")
    kept = [t for t in s.split() if t not in LEGAL_TOKENS and t not in NAME_STOP and t != "dba"]
    return " ".join(kept) if kept else " ".join(toks)


def is_nonlatin_name(raw: str) -> bool:
    """True if the name has letters but none of them are ASCII (Devanagari, Telugu, ...)."""
    has_alpha = False
    for ch in raw:
        if ch.isalpha():
            if ch.isascii():
                return False
            has_alpha = True
    return has_alpha


def _strip0(d: str) -> str:
    s = d.lstrip("0")
    return s or "0"


def addr_forms_v2(raw: str):
    """-> (addr_canon2, postal2, house2, nums2, cids2)
    nums2: ALL digit runs in order (zero-stripped), including runs inside "plotno165", "6th".
    postal2: 5-digit token, or 6-digit token not starting with 0 (India PIN never starts with 0);
             6+ digit tokens starting with 0 are zero-padded house numbers, not postcodes.
    house2: first digit run (<=5 digits after stripping) that is not the postal code.
    cids2: compound identifiers such as "6/79", "8-2-616/1", "bj-99" kept intact."""
    if not raw:
        return "", "", "", "", ""
    low = unicodedata.normalize("NFKD", raw)
    low = "".join(ch for ch in low if not unicodedata.combining(ch)).lower()
    cids = []
    for m in _RE_CID.finditer(low):
        s = re.sub(r"\s+", "", m.group(1))
        cids.append(re.sub(r"(?<!\d)0+(?=\d)", "", s))
    clean = fold(raw)
    out, postal = [], ""
    for t in clean.split():
        if t.isdigit():
            if (len(t) == 5) or (len(t) == 6 and t[0] != "0"):
                postal = t
            t = _strip0(t)
        else:
            t = ADDR_CANON.get(t, t)
            if t in ADDR_STOP:
                continue
        out.append(t)
    if not postal:
        m = re.search(r"\b([1-9]\d{2}) (\d{3})\b", clean)
        if m:
            postal = m.group(1) + m.group(2)
    nums = [_strip0(d) for d in _RE_DIGRUN.findall(clean)]
    house = next((n for n in nums if len(n) <= 5 and n != postal), "")
    return " ".join(out), postal, house, " ".join(nums), " ".join(dict.fromkeys(cids))


def normalize_frame_v2(df):
    """Add v2 columns in place (idempotent)."""
    df["name_key"] = [name_key(x) for x in df["business_name"].values]
    df["name_nonlatin"] = np.fromiter((is_nonlatin_name(x) for x in df["business_name"].values),
                                      dtype=bool, count=len(df))
    af = [addr_forms_v2(x) for x in df["business_address"].values]
    for k, col in enumerate(("addr_canon2", "postal2", "house2", "addr_nums", "addr_cids")):
        df[col] = [x[k] for x in af]
    return df
