#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rollout}"
if [[ "${MODE}" != rollout && "${MODE}" != train ]]; then
  echo "usage: $0 [rollout|train]" >&2
  exit 2
fi

# ---- GPU 数量自动检测 ----
# 优先从 CUDA_VISIBLE_DEVICES 读取，否则从 nvidia-smi 读取
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#_GPU_ARR[@]}"
else
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS="${NUM_GPUS:-1}"
echo "Detected GPUs: ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto})"
# --------------------------

SLIME_DIR="${SLIME_DIR:-/root/slime}"
HF_CHECKPOINT="${HF_CHECKPOINT:-/models/Qwen2.5-Coder-7B-Instruct}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/workspace/models/Qwen2.5-Coder-7B-Instruct_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/examples/coding_agent_rl/local_docker/smoke.jsonl}"
RUN_BASE="${RUN_BASE:-/workspace/runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${RUN_BASE}/${MODE}_qwen25_coder_7b_4gpu_${STAMP}}"

[[ -f "${HF_CHECKPOINT}/config.json" ]] || {
  echo "Missing Hugging Face checkpoint: ${HF_CHECKPOINT}" >&2
  exit 1
}
[[ -s "${PROMPT_DATA}" ]] || {
  echo "Missing prompt data: ${PROMPT_DATA}" >&2
  exit 1
}
if [[ "${MODE}" == train && ! -f "${REF_MODEL_PATH}/latest_checkpointed_iteration.txt" ]]; then
  echo "Missing converted checkpoint: ${REF_MODEL_PATH}" >&2
  echo "Run the four-GPU wrapper in convert mode first." >&2
  exit 1
fi
[[ -f "${SLIME_AGENT_NODE_TARBALL:-}" && -f "${SLIME_AGENT_CC_TARBALL:-}" ]] || {
  echo "SLIME_AGENT_NODE_TARBALL and SLIME_AGENT_CC_TARBALL must point to readable files" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}/rollout_dumps" "${RUN_ROOT}/checkpoints"

archive_ray_logs_on_failure() {
  status=$?
  trap - EXIT
  if [[ "${status}" != 0 && -d /tmp/ray/session_latest/logs ]]; then
    mkdir -p "${RUN_ROOT}/ray_logs"
    cp -aL /tmp/ray/session_latest/logs/. "${RUN_ROOT}/ray_logs/" || true
  fi
  exit "${status}"
}
trap archive_ray_logs_on_failure EXIT

cd "${SLIME_DIR}"
source scripts/models/qwen2.5-7B.sh

export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
export MASTER_PORT=6379
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD:-0.98}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy="localhost,127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST:-slime-coding-agent-main-qwen25-coder-7b-4gpu}"
export NO_PROXY="${no_proxy}"

if [[ "${MODE}" == rollout ]]; then
  N_SAMPLES=1
  DEBUG_ARGS=(--debug-rollout-only)
  CKPT_ARGS=(--hf-checkpoint "${HF_CHECKPOINT}")
else
  N_SAMPLES=4
  DEBUG_ARGS=()
  CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${REF_MODEL_PATH}"
    --load "${RUN_ROOT}/checkpoints"
    --save "${RUN_ROOT}/checkpoints"
    --save-interval 1
  )
fi

ROLLOUT_ARGS=(
  --custom-generate-function-path examples.coding_agent_rl.generate.generate
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --num-rollout 1
  --rollout-batch-size 1
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-context-len "${MAX_CONTEXT_LEN:-32768}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN:-4096}"
  --rollout-temperature 0.6
  --rollout-stop-token-ids 151645
  --num-steps-per-rollout 1
  --global-batch-size "${N_SAMPLES}"
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --calculate-per-token-loss
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

ALGO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --kl-coef 0.0
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

SGLANG_ARGS=(
  --rollout-num-gpus "${NUM_GPUS}"
  --rollout-num-gpus-per-engine "${NUM_GPUS}"
  --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION:-0.70}"
  --sglang-max-running-requests 4
  --sglang-tool-call-parser qwen25
)

MISC_ARGS=(
  --colocate
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address "${MASTER_ADDR}" --num-cpus 16 --num-gpus "${NUM_GPUS}" \
  --object-store-memory 8589934592 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

for attempt in {1..30}; do
  if ray job list --address=http://127.0.0.1:8265 >/dev/null 2>&1; then
    break
  fi
  if [[ "${attempt}" == 30 ]]; then
    echo "Ray dashboard/job agent did not become ready" >&2
    exit 1
  fi
  sleep 2
done

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = (
    "no_proxy", "NO_PROXY", "MASTER_ADDR", "MASTER_PORT",
    "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
    "DOCKER_HOST", "DOCKER_API_VERSION",
    "SWE_AGENT", "SWE_TRAIN_PROTOCOL",
    "SLIME_AGENT_SANDBOX_BACKEND", "SLIME_AGENT_DOCKER_NETWORK",
    "SLIME_AGENT_DOCKER_BINARY", "SLIME_AGENT_DOCKER_EXTRA_RUN_ARGS",
    "SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL",
    "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC",
    "SWE_ROLLOUT_GUARD_SEC", "SWE_BOOT_CONCURRENCY", "SWE_BOOT_RETRIES",
    "ADAPTER_PUBLIC_HOST", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
)
env = {key: os.environ[key] for key in keys if os.environ.get(key)}
env.update({
    "PYTHONPATH": "/root/Megatron-LM:/root/slime",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
})
print(json.dumps({"env_vars": env}))
PY
)"

echo "RUN_ROOT=${RUN_ROOT}"
ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${ALGO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${DEBUG_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/run.log"
