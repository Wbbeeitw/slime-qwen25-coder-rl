#!/usr/bin/env bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/root/slime}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/models/Qwen2.5-Coder-7B-Instruct}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/Qwen2.5-Coder-7B-Instruct_torch_dist}"

if [[ -f "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" ]]; then
  echo "Converted checkpoint already exists: ${REF_MODEL_PATH}"
  exit 0
fi

mkdir -p "${REF_MODEL_PATH}"
cd "${SLIME_DIR}"
source scripts/models/qwen2.5-7B.sh

export PYTHONPATH="/root/Megatron-LM:${SLIME_DIR}:${PYTHONPATH:-}"
python3 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save "${REF_MODEL_PATH}"
