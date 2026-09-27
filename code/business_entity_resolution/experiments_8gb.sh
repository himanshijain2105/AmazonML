#!/usr/bin/env bash
# Staged, evidence-driven experiment plan for an 8 GB Mac. Run ONE stage at a time and read the
# printed comparison table (also saved to $WORK/EXPERIMENTS.md) before running the next one.
#
#   bash experiments_8gb.sh <stage> [DATA_DIR] [WORK_DIR]
#
# Stages:  env | upgrade | sweep | sweep-ablation | train | robustness | final | rehearsal | infer | package
# Tunables (env vars): TOPK (default 40)  MAXDF (5000)  HB (24)  KV (2)  FS (v2)  FINAL_N (50000)
set -euo pipefail
STAGE=${1:?stage}
DATA=${2:-dataset}
WORK=${3:-work}
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
P="python3 -m ber.pipeline"
TOPK=${TOPK:-40}; MAXDF=${MAXDF:-5000}; HB=${HB:-24}; KV=${KV:-2}; FS=${FS:-v2}; FINAL_N=${FINAL_N:-50000}
BLK="--key-version $KV --hash-bits $HB --max-df $MAXDF --top-k $TOPK"
report() { $P report --work "$WORK" | tail -n 40; }

case "$STAGE" in
env)
  python3 --version; python3 -c "import numpy,pandas,scipy,sklearn,lightgbm,rapidfuzz,pyarrow as pa;print('numpy',numpy.__version__,'pandas',pandas.__version__,'lightgbm',lightgbm.__version__,'rapidfuzz',rapidfuzz.__version__,'pyarrow',pa.__version__)"
  (sysctl -n hw.memsize 2>/dev/null | awk '{printf "RAM %.1f GB\n",$1/2^30}') || free -g
  (sysctl -n hw.ncpu 2>/dev/null || nproc) | sed 's/^/CPUs /'; df -h "$WORK" 2>/dev/null || df -h .
  python3 -m pytest -q "$HERE/tests" ;;
upgrade)   # converts the existing v1 cache (work/*.pkl) to slim v2 files; no TSV re-read
  $P renormalize --work "$WORK"
  grep -q EXP-001-baseline "$WORK/experiments.jsonl" 2>/dev/null || cat "$HERE/../../REAL_BASELINES.jsonl" >> "$WORK/experiments.jsonl"
  cat "$WORK/gt_structure.json"; report ;;
sweep)     # SAME 1,000 S1 as the baseline benchmark, FULL pool. Index built once per (keys, hash, max_df).
  $P block-sweep --work "$WORK" --sample-s1 1000 --key-version 1 --hash-bits-list 20 --max-df-list 5000  --top-k-list 40,60,80 --exp-id EXP-002
  $P block-sweep --work "$WORK" --sample-s1 1000 --key-version 1 --hash-bits-list 20 --max-df-list 10000 --top-k-list 40       --exp-id EXP-003
  $P block-sweep --work "$WORK" --sample-s1 1000 --key-version 1 --hash-bits-list 24 --max-df-list 5000  --top-k-list 40       --exp-id EXP-004
  $P block-sweep --work "$WORK" --sample-s1 1000 --key-version 2 --hash-bits-list 24 --max-df-list 5000  --top-k-list 40,60 --exp-id EXP-005 --dump-misses
  python3 "$HERE/tools/analyze_misses.py" "$WORK/misses_EXP-005-v2-h24-df5000-k40.tsv"
  report ;;
sweep-ablation)  # incremental recall of each new v2 key family (drop one family at a time)
  ALL="n,p,c,cp,a,z,nz,nn,an,al,ci"
  for drop in cp nn an,al ci; do
    FAM=$(python3 -c "d=set('$drop'.split(','));print(','.join(f for f in '$ALL'.split(',') if f not in d))")
    $P block-sweep --work "$WORK" --sample-s1 1000 --key-version 2 --families "$FAM" --hash-bits-list 24 \
       --max-df-list 5000 --top-k-list $TOPK --exp-id "EXP-006-minus-${drop//,/}"
  done; report ;;
train)     # same 10K sample and same 2,005 validation S1 as EXP-001 (identical sampler + split hash)
  $P train --work "$WORK" --sample-s1 10000 --key-version 1 --hash-bits 24 --max-df $MAXDF --top-k $TOPK --feature-set v1 --exp-id EXP-010-v1keys-h24-v1feat
  $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set v1 --exp-id EXP-011-v2keys-v1feat
  $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set v2 --exp-id EXP-012-v2keys-v2feat
  $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set v2 --closure --closure-max 80000 --exp-id EXP-013-closure-o2o --dump-val-errors
  $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set v2 --closure --closure-max 80000 --model lr --exp-id EXP-014-logreg
  report ;;
robustness)  # stability across split salts + cross-country transfer (proxy for unseen France)
  for s in a b; do $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set $FS --closure --closure-max 80000 --split-salt $s --exp-id EXP-015-salt-$s; done
  $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set $FS --train-countries US --val-countries India --exp-id EXP-016-US-to-India
  $P train --work "$WORK" --sample-s1 10000 $BLK --feature-set $FS --train-countries India --val-countries US --exp-id EXP-017-India-to-US
  report ;;
final)     # frozen configuration, larger sample; thresholds tuned by THIS run for THIS model
  $P train --work "$WORK" --sample-s1 $FINAL_N $BLK --feature-set $FS --closure --closure-max 150000 --exp-id FINAL --dump-val-errors
  report ;;
rehearsal)
  $P infer --work "$WORK" --model "$WORK/models/FINAL.txt" --out "$WORK/rehearsal_out" --sample-s1 20000 --fresh
  tail -n 1 "$WORK/experiments.jsonl" ;;
infer)     # resumable: re-running continues from the last finished chunk
  $P infer --work "$WORK" --model "$WORK/models/FINAL.txt" --out "$HERE/../../output"
  $P check-output --data "$DATA" --out "$HERE/../../output"
  echo "Now run the OFFICIAL validator:"
  echo "  python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids" ;;
package)
  bash "$HERE/../../make_submission.sh" "${TEAM:?set TEAM=<team name>}" "$DATA" ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
