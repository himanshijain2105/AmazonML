#!/usr/bin/env bash
# Build <team>_submission.zip in the official structure. Refuses if outputs are missing or invalid.
# Usage: bash make_submission.sh <team> <dataset_dir> [path/to/utils/validate_submission.py]
set -euo pipefail
TEAM=${1:?team name}; DATA=${2:?dataset dir}; VAL=${3:-utils/validate_submission.py}
cd "$(dirname "$0")"
for f in output/matching_results.tsv output/candidate_pairs.tsv; do
  [[ -s $f ]] || { echo "REFUSED: $f missing — run 'experiments_8gb.sh infer' first"; exit 1; }
done
PYTHONPATH=code/business_entity_resolution/src python3 -m ber.pipeline check-output --data "$DATA" --out output \
  || { echo "REFUSED: internal output check failed"; exit 1; }
if [[ -f $VAL ]]; then
  python3 "$VAL" --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
     --test-dir "$DATA/test" --check-ids || { echo "REFUSED: official validator failed"; exit 1; }
else
  echo "WARNING: official validator not found at $VAL — run it yourself before uploading"
fi
grep -q "REAL-DATA" Documentation_template.md && echo "WARNING: Documentation_template.md still contains [REAL-DATA] placeholders"
rm -f output/README_GENERATE_OUTPUTS.md
zip -r "${TEAM}_submission.zip" output \
  code/business_entity_resolution/{src,README.md,requirements.txt,experiments_8gb.sh,tools,tests} \
  Documentation_template.md -x '*__pycache__*' '*.pyc'
echo "wrote ${TEAM}_submission.zip"
