"""End-to-end on a small synthetic fixture (plumbing only; not performance evidence)."""
import numpy as np
import pandas as pd
import pytest

from ber import pipeline, synth
from ber.check_output import check


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    synth.generate(str(d / "dataset"), n_s1_train=3000, n_s1_test=800, seed=3)
    w, o = d / "work", d / "output"
    pipeline.main(["prepare", "--data", str(d / "dataset"), "--work", str(w)])
    pipeline.main(["block-sweep", "--work", str(w), "--sample-s1", "300", "--hash-bits-list", "16",
                   "--top-k-list", "20,30", "--exp-id", "T"])
    pipeline.main(["train", "--work", str(w), "--sample-s1", "600", "--top-k", "20", "--hash-bits", "18",
                   "--closure", "--exp-id", "TRAIN"])
    pipeline.main(["infer", "--work", str(w), "--model", str(w / "models" / "TRAIN.txt"), "--out", str(o),
                   "--infer-chunk", "300"])
    return d, w, o


def test_outputs_valid(run):
    d, w, o = run
    assert check(o / "matching_results.tsv", o / "candidate_pairs.tsv", d / "dataset" / "test")


def test_every_test_s1_once_and_france(run):
    d, w, o = run
    m = pd.read_csv(o / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1 = pd.read_csv(d / "dataset/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert m.source1_entity_id.is_unique and set(m.source1_entity_id) == set(s1.entity_id)
    fr = set(s1.entity_id[s1.country == "France"])
    assert fr and (m[m.source1_entity_id.isin(fr)].matched_entity_ids != "").mean() > 0.5


def test_experiment_log_has_all_runs(run):
    d, w, o = run
    import json
    rows = [json.loads(l) for l in open(w / "experiments.jsonl")]
    assert {r["type"] for r in rows} == {"block", "train", "infer"} and len(rows) == 4


def test_resume_is_identical(run):
    d, w, o = run
    before = (o / "matching_results.tsv").read_text()
    (w / "infer_chunks" / "chunk_000000300.npz").unlink()
    pipeline.main(["infer", "--work", str(w), "--model", str(w / "models" / "TRAIN.txt"), "--out", str(o),
                   "--infer-chunk", "300"])
    assert (o / "matching_results.tsv").read_text() == before


def test_features_independent_of_chunking(run):
    """Group features must not depend on how S1 rows are batched (train vs inference batching)."""
    d, w, o = run
    from ber.blocking import Blocker
    from ber.features import FeatureContext, pair_features
    s1, pool = pipeline.load_split(w, "train", pipeline.V1_COLS + pipeline.V2_COLS)
    s1 = s1.head(200).reset_index(drop=True)
    blk = Blocker(hash_bits=16, top_k=20).fit_pool(pool, log=lambda *x: None)
    ctx = FeatureContext(s1, pool, "v2")
    i, j, sn, sa, fb = pipeline.run_blocking(blk, s1)
    whole = pair_features(ctx, i, j, sn, sa, fb)
    parts = []
    for a in range(0, 200, 37):
        m = (i >= a) & (i < a + 37)
        parts.append(pair_features(ctx, i[m], j[m], sn[m], sa[m], fb[m]))
    np.testing.assert_allclose(whole, np.vstack(parts), rtol=1e-6)
