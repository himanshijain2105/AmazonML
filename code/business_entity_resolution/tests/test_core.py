import random

import numpy as np
import pytest

from ber import decide as D
from ber.blocking import blocking_stats, keys_v2
from ber.io import read_source
from ber.metric import f05_entity, fast_macro_f05, full_report, macro_f05
from ber.normalize import addr_forms, addr_forms_v2, deleet, is_nonlatin_name, name_forms, name_key


# ---------- metric (official semantics) ----------
def test_official_example():
    assert f05_entity({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    assert f05_entity(set(), set()) == 1.0 and f05_entity({"S2-1"}, set()) == 0.0 and f05_entity(set(), {"S2-1"}) == 0.0


def test_fast_macro_equals_reference_randomised():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 8)
        gt = {k: [f"S2-{k}-{t}" for t in range(rng.randint(0, 3))] for k in range(n)}
        g, cand, lab, keep = [], [], [], []
        for k in range(n):
            pool = [x for x in gt[k] + [f"S3-x{k}-{t}" for t in range(rng.randint(0, 3))] if rng.random() < 0.8]
            for x in pool:
                g.append(k); cand.append(x); lab.append(x in gt[k]); keep.append(rng.random() < 0.5)
        g, lab, keep = np.array(g, int), np.array(lab, bool), np.array(keep, bool)
        preds = {k: [c for kk, c, kp in zip(g, cand, keep) if kk == k and kp] for k in range(n)}
        n_true = np.array([len(gt[k]) for k in range(n)], float)
        assert fast_macro_f05(g, keep, lab, n_true, n) == pytest.approx(macro_f05(preds, gt))
        assert full_report(g, keep, lab, n_true, n)[0]["macro_f05"] == pytest.approx(macro_f05(preds, gt))


# ---------- v1 normalization (unchanged, audit regressions) ----------
@pytest.mark.parametrize("raw,core", [("Private Eye Investigations LLC", "private eye investigations"),
                                      ("Sharma Traders Private Limited", "sharma traders"), ("Inc", "inc")])
def test_v1_suffix_only_at_end(raw, core):
    assert name_forms(raw)[1] == core


# ---------- v2 normalization: one test per REAL miss pattern ----------
@pytest.mark.parametrize("raw,key", [
    ("Simps0n,Emelita, CPA, (PC)", "simpson emelita cpa"), ("A1egra Limited Private Lal", "alegra lal"),
    ("advanced 1ease inc.", "advanced lease"), ("Melurengineering.Com", "melurengineering"),
    ("#westcollege", "westcollege"), ("www.soliqquest.com", "soliqquest"), ("LLC SILVER NET", "silver net"),
    ("Silver Net LLC", "silver net"), ("Martin Rapid LLC Pineapple", "martin rapid pineapple"),
    ("1st Choice Realty", "1st choice realty"), ("B4 Studios", "b4 studios"), ("Café Élysée SARL", "cafe elysee"),
])
def test_name_key(raw, key):
    assert name_key(raw) == key


def test_deleet_leaves_ordinals_and_alnum():
    assert deleet("1st") == "1st" and deleet("b4") == "b4" and deleet("3m") == "3m" and deleet("we1fare") == "welfare"


def test_nonlatin():
    assert is_nonlatin_name("सनराइज एस्टेट") and not is_nonlatin_name("Premier కన్సల్టెంట్స్") and not is_nonlatin_name("123")


def test_zero_padded_house_is_not_postal():
    a = addr_forms_v2("002121 ADDISON STREET, CHICAGO, IL")
    b = addr_forms_v2("2121 Addison Street, Chicago, IL")
    assert a[0] == b[0] and a[1] == "" and a[2] == b[2] == "2121"
    assert addr_forms("002121 ADDISON STREET")[1] == "002121"   # documents the v1 bug fixed in v2


def test_digit_runs_and_compound_ids():
    canon, postal, house, nums, cids = addr_forms_v2("PLOTNO172 TEACHERS COLONY, ROAD 3")
    assert nums == "172 3" and house == "172"
    assert addr_forms_v2("H.NO 6/79. SECOND FLOOR")[4] == "6/79"
    assert addr_forms_v2("12 Rue de la République, 06000 Nice")[1] == "06000"   # French postcode keeps leading 0


def test_v2_keys_include_address_composites_for_nonlatin_names():
    canon, postal, house, nums, cids = addr_forms_v2("18 Fairview, Mount Ida, AR")
    keys = {k for k, _ in keys_v2("us", "", canon, postal, nums, cids)}
    assert "us|an:18|fairview" in keys and any(k.startswith("us|al:18|") for k in keys)
    assert not any(k.startswith("us|n:") for k in keys)          # empty name key -> address-only retrieval


# ---------- IO ----------
def test_quotes_and_na_are_not_mangled(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                 'S2-1\t"Acme Corp\t1 Main St\tUS\nS2-2\tBeta\t2 Oak\tUS\nS2-3\tGamma" Inc\t3 Elm\tUS\n'
                 "S2-4\tNA\t\tNA\n", encoding="utf-8")
    d = read_source(p)
    assert len(d) == 4 and d.business_name.tolist()[3] == "NA"


# ---------- decisions ----------
def test_decide_zero_one_many_and_o2o():
    g = np.array([0, 0, 0, 1, 2, 2]); cand = np.array([10, 11, 12, 10, 13, 14])
    prob = np.array([.9, .8, .1, .95, .2, .1], np.float32); s3 = np.zeros(6)
    assert D.decide(g, cand, prob, s3, 3, .5, .5).tolist() == [True, True, False, True, False, False]
    assert D.decide(g, cand, prob, s3, 3, .5, .5, 0.0, True).tolist() == [False, True, False, True, False, False]


def test_tune_evaluates_only_eval_groups():
    # group 1 is a competitor: it steals cand 10 under o2o; eval is on group 0 only
    g = np.array([0, 0, 1]); cand = np.array([10, 11, 10]); prob = np.array([.9, .8, .95], np.float32)
    lab = np.array([False, True, True]); n_true = np.array([1.0])
    best = D.tune(g, cand, prob, np.zeros(3), lab, n_true, 2, allow_o2o=True, n_eval=1)
    assert best["o2o"] is True and best["f05"] == pytest.approx(1.0)


def test_blocking_stats_matches_set_reference():
    rng = np.random.default_rng(0)
    i = np.repeat(np.arange(50), 5); j = rng.integers(0, 100, len(i))
    gtl = [rng.choice(100, rng.integers(0, 4), replace=False) for _ in range(50)]
    st, _ = blocking_stats(i, j, 50, gtl)
    cs = set(zip(i.tolist(), j.tolist()))
    tot = sum(len(g) for g in gtl); hit = sum((k, int(x)) in cs for k, g in enumerate(gtl) for x in g)
    assert st["pair_recall"] == pytest.approx(hit / tot)
