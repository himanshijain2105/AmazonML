# Real-data analysis behind the v2 changes

Source: your real Mac artifacts (`work/blocking_benchmark.jsonl`, `work/experiments.jsonl`,
`work/gt_structure.json`, `work/val_errors.tsv` from EXP-001, 2,005 validation S1).
Nothing here comes from synthetic data. No new real runs have been made yet.

## Baseline (real)
| Experiment | Settings | Pair recall | Full-recall entities | Oracle F0.5 | Macro F0.5 | Micro P | Micro R | Singleton acc | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| EXP-000 blocking, 1K S1 | v1 keys, hash_bits 20, max_df 5000, k 40 | 0.8702 | 0.6767 | 0.9466 | – | – | – | – | 2.44 GB |
| EXP-001 train, 10K S1 | same + LightGBM, t=0.68 rel=0.7 o2o=off | 0.8768 | – | 0.9495 | 0.8962 | 0.9734 | 0.7943 | 0.9252 | 2.43 GB |

## Ground truth structure
0 of 7,638,365 matched S2/S3 ids belong to more than one S1 → one-to-one is structurally valid.
EXP-001 could not benefit because validation S1 never competed with the true owners
→ v2 adds competitor-closure validation (`--closure`).

## Blocking misses (861 pairs)
Categories (multi-label): name typo/variant 48.2%, web/hashtag names 22.5%, non-Latin candidate names 18.0%,
legal token not at end 14.6%, zero-padded numbers 12.9%, no shared number 12.5%, different name (DBA-like) 10.8%,
empty candidate address 10.3%, digit-for-letter typos 4.9%.

Share a near-unique key with the true match: **v1 keys 7.1% → v2 keys 88.2%**
(necessary for retrieval, not sufficient — the real sweep must confirm).
Specific findings: every real run used hash_bits=20 (~1M buckets for 10.3M records → collisions);
zero-padded house numbers were parsed as postal codes; US S1 addresses carry no ZIP, so name×postal
matched 0% of misses.

## False positives (144 + 9 singleton)
Mostly near-duplicate hard negatives differing only in a number ("415 vs 417 Mccurdy Ave",
"PLOTNO165 vs PLOTNO172", "6th vs 8th Flr-602") → v2 digit-run conflict features; plus identical names
with empty/thin addresses, likely owned by another S1 → one-to-one via closure.
Non-Latin candidate names appear in 22–24% of scored errors → cross-script flags.
