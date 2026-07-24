#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|test|sandbox-smoke|claude-smoke|convert|rollout|train) ;;
  *)
    echo "usage: $0 [preflight|test|sandbox-smoke|claude-smoke|convert|rollout|train]" >&2
    exit 2
    ;;
esac

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKSPACE="${SLIME_CODING_AGENT_WORKSPACE:-/data/xys/slime-coding-agent}"
MODEL_DIR="${QWEN25_CODER_7B_DIR:-/data/slime_qwen_coder/models/Qwen2.5-Coder-7B-Instruct}"
RUNTIME_IMAGE="${SLIME_CODING_AGENT_IMAGE:-slime-coding-agent-runtime:local}"
TASK_IMAGE="${SLIME_AGENT_TASK_IMAGE:-slime-coding-agent-swe-smoke:local}"
NETWORK="${SLIME_AGENT_DOCKER_NETWORK:-slime-coding-agent-net}"
CONTAINER_NAME="${SLIME_LOCAL_MAIN_CONTAINER:-slime-coding-agent-main-qwen25-coder-7b-4gpu}"
GPU_DEVICE="${SLIME_CODING_AGENT_GPU_DEVICE:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
CONTAINER_CUDA_VISIBLE_DEVICES="${SLIME_CONTAINER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
if [[ "${GPU_DEVICE}" == *,* ]]; then
  DOCKER_GPU_REQUEST="\"device=${GPU_DEVICE}\""
else
  DOCKER_GPU_REQUEST="device=${GPU_DEVICE}"
fi

preflight() {
  command -v docker >/dev/null
  command -v nvidia-smi >/dev/null
  [[ -f "${MODEL_DIR}/config.json" ]] || {
    echo "Missing Qwen2.5-Coder-7B model: ${MODEL_DIR}" >&2
    exit 1
  }
  [[ -f "${WORKSPACE}/assets/node-v22-linux-x64.tar.xz" ]] || {
    echo "Missing Node asset under ${WORKSPACE}/assets" >&2
    exit 1
  }
  [[ -f "${WORKSPACE}/assets/anthropic-ai-claude-code.tgz" ]] || {
    echo "Missing Claude Code asset under ${WORKSPACE}/assets" >&2
    exit 1
  }
  docker image inspect "${RUNTIME_IMAGE}" >/dev/null
  docker image inspect "${TASK_IMAGE}" >/dev/null
  docker network inspect "${NETWORK}" >/dev/null 2>&1 || \
    docker network create "${NETWORK}" >/dev/null
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo "Qwen2.5-Coder-7B four-GPU local Docker preflight: OK"
}

preflight
if [[ "${MODE}" == preflight ]]; then
  exit 0
fi

if [[ "${MODE}" == convert ]]; then
  docker run --rm \
    --name "${CONTAINER_NAME}-convert" \
    --gpus "${DOCKER_GPU_REQUEST}" \
    --ipc=host \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit nofile=1048576:1048576 \
    --ulimit stack=67108864 \
    --env CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}" \
    --volume "${ROOT_DIR}:/root/slime" \
    --volume "${WORKSPACE}:/workspace" \
    --volume "${MODEL_DIR}:/models/Qwen2.5-Coder-7B-Instruct:ro" \
    --workdir /root/slime \
    "${RUNTIME_IMAGE}" \
    bash examples/coding_agent_rl/local_docker/convert_qwen25_coder_7b.sh
  exit 0
fi

COMMON_ARGS=(
  --rm
  --name "${CONTAINER_NAME}"
  --network "${NETWORK}"
  --gpus "${DOCKER_GPU_REQUEST}"
  --ipc=host
  --shm-size=32g
  --ulimit memlock=-1
  --ulimit nofile=1048576:1048576
  --ulimit stack=67108864
  --env CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_VISIBLE_DEVICES}"
  --env SWE_AGENT=claude_code
  --env SWE_TRAIN_PROTOCOL=scaleswe
  --env SLIME_AGENT_SANDBOX_BACKEND=docker
  --env SLIME_AGENT_DOCKER_NETWORK="${NETWORK}"
  --env SLIME_AGENT_DOCKER_BINARY=/usr/local/bin/docker
  --env SLIME_AGENT_NODE_TARBALL=/workspace/assets/node-v22-linux-x64.tar.xz
  --env SLIME_AGENT_CC_TARBALL=/workspace/assets/anthropic-ai-claude-code.tgz
  --env ADAPTER_PUBLIC_HOST="${CONTAINER_NAME}"
  --env ADAPTER_BIND_HOST=0.0.0.0
  --env ADAPTER_PORT=18001
  --env SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-300}"
  --env SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-120}"
  --env SWE_BOOT_CONCURRENCY=1
  --env SWE_BOOT_RETRIES=1
  --env SLIME_AGENT_CC_EXTRA_ARGS="--disable-slash-commands --disallowedTools WebFetch WebSearch"
  --volume "${ROOT_DIR}:/root/slime"
  --volume "${WORKSPACE}:/workspace"
  --volume "${MODEL_DIR}:/models/Qwen2.5-Coder-7B-Instruct:ro"
  --volume /var/run/docker.sock:/var/run/docker.sock
  --workdir /root/slime
)

case "${MODE}" in
  test)
    docker run "${COMMON_ARGS[@]}" "${RUNTIME_IMAGE}" pytest -q tests/test_agent
    ;;
  sandbox-smoke)
    docker run "${COMMON_ARGS[@]}" "${RUNTIME_IMAGE}" \
      python3 examples/coding_agent_rl/local_docker/sandbox_smoke.py
    ;;
  claude-smoke)
    docker run "${COMMON_ARGS[@]}" "${RUNTIME_IMAGE}" \
      python3 examples/coding_agent_rl/local_docker/claude_smoke.py
    ;;
  rollout|train)
    docker run "${COMMON_ARGS[@]}" "${RUNTIME_IMAGE}" \
      bash examples/coding_agent_rl/local_docker/run_qwen25_coder_7b_4gpu.sh "${MODE}"
    ;;
esac
