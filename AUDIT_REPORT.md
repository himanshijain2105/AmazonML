# Audit Report — Antigravity implementation, Amazon ML Challenge 2026 (Business Entity Resolution)

**Scope.** Every file in `amazon-ml-entity-resolution.zip` was read (13 modules in `src/`, 7 test files, 2 audit
artifacts, empty `data/ models/ notebooks/ output/`, `__MACOSX` junk). There is **no README, no
requirements.txt, no trained model, no output, no experiment log**. The official README, template and validator
were treated as authoritative.

**Evidence standard.** The real challenge files were not available in the review environment
(1 CPU, 3 GB RAM). Every number below is either (a) taken from the project's own `data_audit.md`, (b) measured
by running the original code on a synthetic dataset that reproduces the audited structure
(multi-match distribution, 5.6% singletons, ~26% unmatched distractors, heavy duplicate S1 names from chains,
3.4% empty addresses, heavier S2 punctuation, France only in test), or (c) marked **NOT YET MEASURED**.
Synthetic F0.5 values are meaningful only *relative to each other*; they are **not** estimates of leaderboard scores.

---

## 1. Executive verdict

The original project has a correct metric, a correct S1-level split and correctly formatted writers, but it
**cannot produce a full-scale submission**: blocking is unbounded (≈4,900 candidates per S1 measured on a
94K-record pool), the in-memory object model needs ≈40 GB for the training S2/S3 pool before any candidate is
generated, the standalone `train.py`/`inference.py` fail with the project's own feature function, and the pilot
mode inflates recall through a label-sampled pool.

| Dimension | Score | Reason |
|---|---|---|
| Competition readiness | **2/10** | No working path from data to a valid full-scale submission; `--mode train/validate/infer` not implemented; OOM at 20K S1 × 94K pool. |
| Code quality | **5/10** | Readable, documented, correct metric, 81 unit tests pass; but duplicated logic (pipeline vs train/inference/thresholding), incompatible interfaces, dead code, two conflicting abbreviation tables. |
| Scalability | **1/10** | Unbounded candidates; ~3.9 KB/record Python heap (measured) ⇒ tens of GB; per-pair Python lists for training. |
| ML methodology | **3/10** | Good: S1-level split, threshold chosen on macro F0.5. Bad: leaky pilot pool, default class re-weighting, threshold tuned on a different model than the one shipped, no chain/competition features, independent per-pair thresholding. |
| Challenge compliance | **6/10** | Output semantics right (every S1, blank singleton lists, candidates = scored set, no external data, open-set country string); missing README/requirements/doc; not reproducible end to end. |

**Recommendation: MAJOR REFACTOR REQUIRED** — delivered in `code/business_entity_resolution/`
(details §13).

---

## 2. What is genuinely COMPLETE (original code)

- **Official metric** (`evaluation.py`): per-S1 F0.5, singleton rule (empty→1.0, any→0.0), macro over *all*
  ground-truth S1 (missing predictions = empty). Cross-checked against an independent implementation on
  20,000 random cases: **max difference 0**. Computes no pairwise F1 / micro / AUC in place of the official metric
  (micro metrics are only reported as extras).
- **S1-level train/validation split** (`validate.py`), reproducible.
- **Output writers**: exact headers, tab separated, one row per S1, sorted, blank lists for no-match, comma-joined
  without spaces; the candidate file written is the set that was scored (matches ⊆ candidates holds in `pipeline.py`).
- **Loaders** in `pipeline.py`/`io_utils.py` split lines manually and are therefore immune to the pandas
  quote-merging trap (see H11) — credit where due.
- **Streaming data audit** (`data_audit.py`) and its artifacts.
- **Country as an open-set string** (`normalize_country`); no hard-coded country filter in blocking/features.

## 3. What is PARTIALLY complete

| Module | Implemented | Validated? |
|---|---|---|
| normalization.py | yes | Unit tests only; suffix stripping is wrong (C5 → H1), expanded address never used (H2). |
| blocking.py | yes | Never measured on real data; measured here as unbounded. |
| features.py | 32 features | Correct per pair, but Python-dict-per-pair; missing key signals (H6). |
| train.py | LR + XGB | Fails with project's own feature function (C5). |
| thresholding.py | global + per-source | Per-source objective is not the official metric (M1); unused by pipeline. |
| inference.py | yes | Fails with project's own feature function (C5); unused by pipeline. |
| error_analysis.py | 5 categories | Cannot distinguish blocking misses; not wired into the pipeline. |
| pipeline.py | sample, full | Sample mode leaky (C4); full mode infeasible (C1–C3); other modes stubs. |

## 4. What is NOT DONE

README; requirements.txt; any run on real data (no blocking recall, no F0.5, no runtime, no memory:
**all NOT YET MEASURED**); train/infer separation with a saved model; hard-negative/candidate-volume control;
France/transfer validation; competition-aware decisions; integration tests; methodology document; final package.

---

## 5. CRITICAL ISSUES

**C1 — Unbounded candidate generation** (`blocking.py`, `generate_candidates`). Any S2/S3 record sharing a single
name token with df ≤ 5,000 and the same country becomes a candidate; there is no per-S1 cap
(`top_k_candidates`/`max_candidates_per_entity` in `config.py` are never read). Measured (synthetic, 1,500 S1 vs
93,837 S2/S3): **mean 4,945, median 4,745, p95 8,555, max 10,836 candidates per S1** (5.3% of the whole pool per S1),
recall 0.992. Each index posting list can reach 5,000 per token, so at 10M-record pools the product is on the order
of 10⁹–10¹⁰ pairs. Infeasible.

**C2 — Object-heavy in-RAM data model.** Every record is a Python dict with 17 keys + token lists; indexes are
`defaultdict(set)` of strings. Measured with tracemalloc: **2,340 B/record** (normalized records) +
**1,561 B/record** (indexes) ⇒ ≈24 GB + 16 GB for the training S2/S3 pool alone. `run_full_pipeline` keeps the
train engine and dicts alive while loading and indexing test ⇒ well above 80 GB. The original pipeline was
**OOM-killed at 20K S1 × 94K S2/S3** in a 3 GB environment.

**C3 — Training matrix materialised as Python lists** (`prepare_training_pairs`, `train.prepare_training_data`).
List-of-lists of Python floats ≈ 1 KB per pair before `np.array`; even at a sane 40 candidates/S1, 1.76M train S1
⇒ ~70M pairs ⇒ ~70 GB. With C1's candidate volume it is impossible.

**C4 — Label-leaking pilot mode** (`run_sample_experiment`). The S2/S3 pool is *all true matches of the sampled S1*
plus 10× random rows. Every true match is guaranteed present and distractors are 150× rarer than in the real pool,
and IDF/df caps are calibrated on the tiny pool. Measured (synthetic, 5,000 S1): blocking recall **0.9887 on the
full pool vs 0.9965 on the label-sampled pool** (misses cut by ~70% from a pool only 6× smaller). Any sample-mode
F0.5 is optimistic and must not drive decisions.

**C5 — Broken module interfaces.** `compute_pair_features` returns a dict; `train.prepare_training_data` and
`inference.run_inference` expect a vector. Reproduced: `TypeError: float() argument must be ... not 'dict'` and
`ValueError: Please reshape the input data into 2-dimensional matrix`. `pipeline.py` re-implements both inline, so the
tested modules are not the ones that run; `--mode train/validate/infer` print "not yet implemented", so inference
from a saved model is impossible.

## 6. HIGH-PRIORITY ISSUES

**H1 — Legal suffixes stripped anywhere in the name** (`_RE_LEGAL_SUFFIXES` has no end anchor). Reproduced:
"Private Eye Investigations LLC"→"eye investigations", "AG Motors Inc"→"motors", "Limited Edition
Boutique"→"edition boutique", "Co-op Grocery"→"-op grocery". The result is an exact blocking key and a feature —
false merges under a precision-weighted metric.

**H2 — Address canonicalisation unused.** `addr_expanded` is computed but never used by blocking or features;
reproduced `normalized_addr_exact=0` for "1 Main St" vs "1 Main Street". `config.ADDRESS_ABBREVIATIONS`
(~150 entries) and `normalization.ADDRESS_ABBREVIATIONS` (12 entries) disagree; the larger one is dead code.

**H3 — Addresses barely used in blocking.** Only an exact full-address key; `addr_token_index` is built (memory)
but never queried. Names whose tokens are all frequent (chains) can only be found by exact name keys.

**H4 — Name and address token frequencies share one df table**, so df caps for name tokens are distorted by address
occurrences (e.g. "park", "gandhi", "main").

**H5 — Threshold/model mismatch and class re-weighting.** Thresholds are tuned on the 80%-split model trained with
`scale_pos_weight=neg/pos`, then the shipped model is retrained on 100% with a different weight; the probability
scale moves and the tuned threshold is not guaranteed valid. Re-weighting also decalibrates probabilities; measured
(synthetic): `scale_pos_weight=9` gave 0.9915 vs 0.9921 unweighted and pushed the optimum to the grid edge (0.95).

**H6 — Missing discriminative features.** No IDF-weighted similarity, no name-commonness (chain risk) signal, no
postal/house-number equality **or conflict**, no blocking score, no within-S1 competition features.
`name_strong_addr_weak` actually tests name > 0.9 **and** address > 0.5 (misnamed). Several exact flags are redundant.

**H7 — Independent per-pair thresholding.** No relative-to-best rule and no competition across S1 entities for the
same S2/S3 record. With ~30% duplicate S1 names this is the main precision lever and it is absent. Whether each
S2/S3 id belongs to at most one S1 is **NOT YET MEASURED** on real data (`gt_structure.json` in the new pipeline).

**H8 — No France/transfer control.** Measured (synthetic): training on US only and validating on India gives
**0.936** vs **0.990** when India is in training — the transfer risk is real and unmonitored.

**H9 — Rule baseline predicts every same-country exact-name record** (`run_rule_baseline`), i.e. every chain branch;
it is labelled high-precision but is the opposite.

**H10 — Not reproducible**: no README/requirements; 8 tests hard-require the real dataset at a fixed relative path.

**H11 — Trap to avoid when refactoring**: the official README suggests `pd.read_csv(sep="\t")`. Reproduced: a
name starting with `"` silently merged **4 rows into 2**, and a business named `NA` became NaN. Use
`quoting=QUOTE_NONE, keep_default_na=False, dtype=str` and a row-count check.

## 7. MEDIUM / LOW ISSUES

- **M1** `optimize_source_specific_thresholds` optimises S2 and S3 thresholds on *per-source* macro F0.5 (an S1
  with only S3 matches counts as an S2 "singleton") — not the official objective. Duplicated threshold code in
  `pipeline.py`.
- **M2** Threshold grids stop at 0.95/0.85 — optimum can sit on the edge.
- **M3** `rapidfuzz` silently falls back to `difflib` (~100× slower); unpinned.
- **M4** `evaluate_blocking` reduction ratio uses "targets seen in candidates" as the pool size when not given — misleading.
- **M5** Error analysis cannot separate blocking misses from classifier misses; no chain / same-address categories.
- **M6** `load_source_streaming` pads short rows silently (hides malformed lines); no row-count check.
- **L1** `__MACOSX` files in the archive; empty directories; `use_label_encoder` deprecated; duplicated
  `country_norm`/`country_normalized` keys; `n_workers` parameters ignored.

---

## 8. Scalability assessment

| Workload | Original code | Refactored code |
|---|---|---|
| 2.2M train S1 | **No** (C1–C3) | Yes with `--sample-s1 500k–800k` (recommended); all 2.2M ⇒ ~77M pairs × 34 × 4 B ≈ 10.5 GB features, needs ≥64 GB |
| 10M+ train S2/S3 | **No** (~40 GB heap before candidates) | Yes: pool held as pandas columns + sparse matrices |
| 1.73M test S1 | **No** | Yes, ≤40 candidates/S1 by construction (~60–70M pairs), chunked |
| 10M+ test S2/S3 | **No** | Yes; estimated persistent ≈18–20 GB, peak ≈25–30 GB (**extrapolated from synthetic; NOT YET MEASURED**) |

Measured for the refactor (synthetic, 1 CPU): 80K S1 vs 376K pool → 2.52M pairs, blocking + all features ≈42 s,
peak RSS 1.99 GB; persistent memory 1,076 B/record (DataFrames), 374 B/record (feature context), blocker ≈0.55 GB
fixed + ~100 B/record. Per-S1 blocking cost grows with posting-list length, which is capped by `max_df`, and peak
memory per chunk is capped by `--nnz-budget`; **real runtime at 10M pools is NOT YET MEASURED** — measure it with
`block-stats --sample-s1 20000` and `infer --sample-s1 20000` before the full run.

## 9. Architecture assessment

**Original**
```
TSV → list[dict] (all rows, all sources) → normalize_record (13 fields/record)
    → BlockingEngine: 7 dict-of-set indexes; exact keys ∪ rare-name-token hits (unbounded)
    → per-pair compute_pair_features (dict) → Python list-of-lists → XGBoost (scale_pos_weight)
    → per-pair threshold tuned on val → retrain on 100% → rebuild engine on test → write TSVs
    (train.py / inference.py / thresholding.py / error_analysis.py: parallel, unused or broken)
```
**Refactored (delivered)**
```
TSV (quote/NA-safe, row-count check) → pandas columns + normalized forms (cached .pkl)
  → Blocker.fit_pool(S2∪S3 of the split): hashed keys, pool IDF, df cap, 2 weighted channels
  → query S1 in nnz-budgeted chunks: top-24 name ∪ top-16 address   [≤40 cand/S1 = candidate_pairs.tsv]
  → pair_features (vectorised, 34 feats, identical code for train & test)
  → LightGBM (early stopping on held-out S1) → probabilities
  → decide(): p ≥ t_src, p ≥ rel·max_p(S1), optional one-to-one per S2/S3 record
      (all tuned on validation macro F0.5 incl. singletons and blocking misses)
  → matching_results.tsv ⊆ candidate_pairs.tsv → check-output + official validator
```

## 10. Experiment plan (commands refer to the refactored pipeline; all results NOT YET MEASURED on real data)

| ID | Goal | Input / command | Measurement | Success criterion | Runtime concern |
|---|---|---|---|---|---|
| EXP-001 | Exact baseline | `baseline --sample-s1 50000` | macro F0.5 | establishes floor | pool indexing only |
| EXP-002 | Fuzzy rule | same run | macro F0.5 (in-sample tuned, optimistic) | model must beat it clearly | features on 50K S1 |
| EXP-003 | Blocking + simple scorer | `block-stats` + EXP-002 rule | recall, ceiling | ceiling − rule gap = model headroom | — |
| EXP-004 | Linear model check | train with LR swapped in `model.py` (optional) | macro F0.5 | only to confirm GBDT gain is real | low |
| EXP-005 | LightGBM | `train --sample-s1 100000` | val macro F0.5, ceiling gap, per country, singleton acc | beats EXP-002; gap to ceiling shrinks | ~3.5M pairs |
| EXP-006 | Blocking sweep | `block-stats --sample-s1 20000 --top-k {25,40,60} --max-df {2000,5000,20000}` | pair recall, full-recall entities, ceiling, s/S1, peak RSS | ceiling ≥ ~0.99 at ≤40 cands; s/S1 × 1.73M fits budget | the key timing measurement |
| EXP-007 | Decision rule | automatic in `train` | f05 for global / rel / o2o | keep only rules with gain | seconds |
| EXP-008 | Hard negatives / volume | top-k 25 vs 40 vs 60 in `train` | macro F0.5 vs pairs | stop increasing k when F0.5 flat | linear in k |
| EXP-009 | Source thresholds | automatic (≥5e-4 gain gate) | f05 vs global_rule_f05 | accept only above gate | seconds |
| EXP-010 | Final | `train --sample-s1 600000` → `infer` → validators | val F0.5, per-country match-rate on test | validators PASS; France match-rate/empty-rate in line with US/India | full test |
| EXP-X | France proxy | `train --train-countries US --val-countries India` (and reverse) | transfer drop | informs how much to trust France | as EXP-005 |

Development reference points (synthetic, 20K S1, NOT real): EXP-001 0.630 · EXP-002 0.895 · single-channel
blocking + LightGBM 0.982 (ceiling 0.988) · two-channel 0.992 (ceiling 0.998) · US→India transfer 0.936.

## 11. Required fixes BEFORE full-scale experiments

For the original code (all implemented in the refactor):
1. Bound candidates per S1 (top-k from a scored retrieval), and make that list the scored list and the candidate file (C1).
2. Replace dict-per-record + set indexes with columnar arrays + sparse matrices (C2).
3. Stream features in S1 chunks into float32 arrays; never build Python lists of pairs (C3).
4. Pilots must use the full S2/S3 pool; sample only S1 (C4).
5. One feature function, returning arrays, shared by training and inference; a saved model with a feature-name check (C5).
6. End-anchored legal-suffix stripping; canonical address forms used by both blocking and features (H1, H2).
7. Tune thresholds for the model that is shipped; no default class re-weighting (H5).

For the refactor, before the full run on real data:
8. `prepare` must pass the row-count checks; read `gt_structure.json` (one-to-one or not).
9. `block-stats --sample-s1 20000` on the real pool: record s/S1, peak RSS, recall, ceiling; set `top_k`, `max_df`, `nnz-budget`.
10. Confirm RAM ≥32 GB (64 GB for comfort) and run `infer --sample-s1 20000` as a timing rehearsal.

## 12. Safe-to-run checklist

| Question | Original code | Refactored code |
|---|---|---|
| 20K S1 pilot? | **NO** — loads all 12.5M train records as dicts first (~9 GB+), then unbounded candidates (OOM measured at 20K×94K), and results are leaky (C4). | **YES** on a ≥32 GB machine (full train pool is loaded: ~11 GB DataFrames + ~5 GB structures, estimated). |
| 100K S1 pilot? | **NO** — same reasons, 5× the pairs. | **YES** (≈3.5M pairs, features ≈0.5 GB) after the 20K pilot's RSS is read. |
| Full training set? | **NO** | **YES with `--sample-s1 500k–800k`**; all 2.2M only with ≥64 GB (not needed for a 34-feature GBDT). |
| Full test inference? | **NO** — no inference mode, and C1/C2 apply. | **YES, conditionally**: after `infer --sample-s1 20000` confirms time and peak RSS extrapolate within budget (peak estimated 25–30 GB, NOT YET MEASURED). |

## 13. Final recommendation — **MAJOR REFACTOR REQUIRED**

Not a rebuild: the metric, the S1-level split idea, output semantics, quote-safe loading and the data audit are
correct and were kept (metric re-implemented and verified identical). But the three load-bearing parts — the
data model, blocking, and training-data preparation — are structurally unable to run at 2.2M/10M scale, and the
decision layer ignores the dominant precision risk (chains/same-name businesses). These were replaced; the
replacement was verified end to end on synthetic data (21 tests, both validators PASS with `--check-ids`,
France handled), and it has **not yet been run on the real data**.

---

## The 5 things to fix or verify FIRST before spending more compute

1. **Retire the original blocking engine and pilot mode, and re-measure blocking on the real full pool.**
   `BlockingEngine.generate_candidates` is unbounded (~4,900 candidates/S1 measured on a 94K pool) and
   `run_sample_experiment` inflates recall with a label-sampled pool. Run
   `block-stats --sample-s1 20000` with top-k 25/40/60 and max-df 2k/5k/20k; pick the setting from pair recall,
   the oracle macro-F0.5 ceiling, seconds per S1 and peak RSS. Every later number is bounded by this.
2. **Check `gt_structure.json` after `prepare`.** If no S2/S3 id belongs to more than one S1, the one-to-one
   decision is enabled automatically and is the main defence against chain false merges (1.54M distinct names across
   2.21M S1). If it is not one-to-one, the rule stays off and chain precision relies on the group features — then
   inspect `FP` rows in `val_errors.tsv` first.
3. **Keep the fixed normalization and verify it on real strings.** The original suffix regex deletes legal words
   anywhere in the name and the canonical address is never used; sample 1,000 real S2 names and addresses and
   eyeball `name_core`, `addr_canon`, `postal`, and check that the input files load with the row-count check
   (look for `"` in names: `grep -c '"' train_source2.tsv`).
4. **Quantify the France risk before trusting the submission.** Run `train --train-countries US --val-countries India`
   and the reverse. After test inference, compare per-country predicted-match rate and empty-row rate (logged by
   `infer`): France should look like US/India (≈3.4 matches/S1, ≈5–6% empty). A large deviation means the threshold
   does not transfer.
5. **Tune thresholds for the exact model you ship, on macro F0.5, without class re-weighting.** The original tunes
   on the 80% model with `scale_pos_weight=neg/pos` and then ships a retrained 100% model with a different weight.
   The refactor ships the validated model with its tuned decision; if you refit on more data, re-tune on a fresh
   held-out S1 slice before inference.
