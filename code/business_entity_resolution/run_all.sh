#!/usr/bin/env bash
# End-to-end reproduction: data -> normalization -> blocking -> features -> model -> outputs.
# Usage: bash run_all.sh /path/to/student_resource/dataset [WORK_DIR] [OUTPUT_DIR] [TRAIN_S1_SAMPLE]
set -euo pipefail
DATA=${1:?path to dataset/ (containing train/ and test/)}
WORK=${2:-work}
OUT=${3:-output}
SAMPLE=${4:-600000}          # S1 entities used for training+validation (0 = all 2.2M; needs ~64GB+)
HERE="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
python3 -m ber.pipeline prepare      --data "$DATA" --work "$WORK"
python3 -m ber.pipeline block-stats  --work "$WORK" --sample-s1 20000
python3 -m ber.pipeline train        --work "$WORK" --sample-s1 "$SAMPLE" --exp-id FINAL --dump-val-errors
python3 -m ber.pipeline infer        --work "$WORK" --out "$OUT"
python3 -m ber.pipeline check-output --data "$DATA" --out "$OUT"
