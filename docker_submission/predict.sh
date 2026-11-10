#!/bin/bash
# Entrypoint used by the organizers: /bin/bash /code/predict.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=2026
cd "$(dirname "$0")" || exit 1
START=$(date +%s)
python3 preprocessing.py || echo "preprocessing reported a problem; continuing" >&2
python3 predict.py
STATUS=$?
echo "TOTAL_SECONDS=$(( $(date +%s) - START )) exit=$STATUS"
exit $STATUS
