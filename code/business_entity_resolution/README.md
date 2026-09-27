# Business Entity Resolution — Amazon ML Challenge 2026 (8 GB-safe, v2)

Blocking (compact inverted index, two channels) → LightGBM pair model → macro-F0.5-tuned decision
(threshold, relative-to-best, optional one-to-one). Only challenge data; no external data/APIs;
LightGBM (MIT), scikit-learn/scipy/numpy/pandas (BSD), rapidfuzz (MIT), pyarrow (Apache-2.0).

## Run on the Mac (one stage at a time; read the printed table before the next)
```bash
pip install -r requirements.txt
bash experiments_8gb.sh env
bash experiments_8gb.sh upgrade   dataset work   # converts existing v1 cache to slim v2 (no TSV re-read)
bash experiments_8gb.sh sweep     dataset work   # EXP-002..005 on the SAME 1,000 S1 as the baseline
bash experiments_8gb.sh train     dataset work   # EXP-010..014 on the SAME 2,005 val S1 as EXP-001
bash experiments_8gb.sh robustness dataset work  # split salts, US->India, India->US
TOPK=.. MAXDF=.. bash experiments_8gb.sh final dataset work
bash experiments_8gb.sh rehearsal dataset work   # 20K test S1, full pool: timing + RSS
bash experiments_8gb.sh infer     dataset work   # resumable full inference + output check
TEAM=myteam bash experiments_8gb.sh package dataset work
```
Fresh start without a cache: `python -m ber.pipeline prepare --data dataset --work work`.
Every run appends to `work/experiments.jsonl`; `python -m ber.pipeline report --work work` writes `EXPERIMENTS.md`.
Any stage stops cleanly if peak RSS exceeds `--max-rss-gb` (default 6.5).

## Versions kept side by side
`--key-version 1|2` (v1 reproduces the baseline blocker, 99.8% pair Jaccard on the fixture) and
`--feature-set v1|v2`, so every change is measured against the baseline on identical samples.
See `../../REAL_ANALYSIS.md` for the real-data evidence behind v2.

## Tests
`python -m pytest -q tests` (32 tests; synthetic fixture = plumbing only, never performance evidence).
`tools/analyze_misses.py` categorises real blocking misses dumped by the sweep.
