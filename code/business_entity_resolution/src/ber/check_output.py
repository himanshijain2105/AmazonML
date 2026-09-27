"""Submission checks mirroring the official validator, PLUS the checks it does not do:
matches ⊆ candidates is enforced as an error (not a warning), and IDs are verified to exist
in the test S2/S3 files. Streams files; memory ~ size of the ID sets."""
from __future__ import annotations

import sys
from pathlib import Path


def _ids(path):
    with open(path, encoding="utf-8") as f:
        next(f)
        return {l.split("\t", 1)[0] for l in f if l.strip()}


def _read(path, header):
    errs, m = [], {}
    with open(path, encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        if h != header:
            return None, [f"{path.name}: header {h} != {header}"]
        for n, line in enumerate(f, 2):
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                errs.append(f"{path.name}:{n} no tab")
                continue
            ids = rest.split(",") if rest else []
            if s1 in m:
                errs.append(f"{path.name}: duplicate row {s1}")
            if len(ids) != len(set(ids)):
                errs.append(f"{path.name}: duplicate id in list for {s1}")
            m[s1] = set(ids)
    return m, errs


def check(matching, candidate, test_dir) -> bool:
    test_dir = Path(test_dir)
    req = _ids(test_dir / "test_source1.tsv")
    valid = _ids(test_dir / "test_source2.tsv") | _ids(test_dir / "test_source3.tsv")
    errs = []
    M, e = _read(Path(matching), ["source1_entity_id", "matched_entity_ids"]); errs += e
    C, e = _read(Path(candidate), ["source1_entity_id", "candidate_entity_ids"]); errs += e
    for name, d in (("matching", M), ("candidate", C)):
        if d is None:
            continue
        if set(d) != req:
            errs.append(f"{name}: S1 rows differ from test_source1 (missing {len(req - set(d))}, extra {len(set(d) - req)})")
        bad = {x for v in d.values() for x in v if x not in valid}
        if bad:
            errs.append(f"{name}: {len(bad)} ids not in test S2/S3 (e.g. {sorted(bad)[:3]})")
    if M is not None and C is not None:
        viol = [k for k, v in M.items() if v - C.get(k, set())]
        if viol:
            errs.append(f"{len(viol)} S1 have matches not in candidate_pairs (e.g. {viol[:3]})")
    for x in errs[:20]:
        print("ERROR:", x)
    print("PASS" if not errs else f"FAIL ({len(errs)} issues)")
    return not errs


if __name__ == "__main__":
    sys.exit(0 if check(*sys.argv[1:4]) else 1)
