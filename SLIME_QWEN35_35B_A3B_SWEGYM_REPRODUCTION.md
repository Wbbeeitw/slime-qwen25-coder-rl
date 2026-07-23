# slime Qwen3.5-35B-A3B + SWE-Gym 双服务器训练复现手册

## 1. 文档目标

本文复现已经在 2026-07-18 完整跑通的 Agent Code RL 单步训练：

```text
Qwen3.5-35B-A3B
  -> 双 H200 上的 SGLang Agent rollout
  -> 192.168.110.101 上的 SWE-Gym Docker task/eval sandbox
  -> SWE-Bench 风格官方 grader
  -> GRPO batch
  -> Megatron forward/backward
  -> StatelessAdam + file-backed FP32 master
  -> iteration 0 checkpoint
```

本文不会覆盖以下已验证入口和文档：

```text
SLIME_LOCAL_REPRODUCTION.md
SLIME_REMOTE_DOCKER_REPRODUCTION.md
SLIME_REMOTE_DOCKER_35B_A3B_REPRODUCTION.md
examples/coding_agent_rl/local_docker/run_host.sh
examples/coding_agent_rl/local_docker/run_remote_docker.sh
examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh
```

本文默认复现的是一个真实 SWE-Gym task、两个 trajectory、一个训练 step。它验证完整训练
链路，但不是 SWE-Gym 2438 条任务的全量 epoch。

## 2. 复现范围和版本边界

当前 slime 工作树基于：

```text
/home/xys/slime
base commit: fb42ae456fac8166afb604f13b30d22bb3c75053
```

SWE-Gym 兼容代码、remote Docker 支持和 35B 内存 hook 目前仍包含本地未提交改动。因此本文
首先保证“在当前两台服务器上重复运行”可复现；不能只 checkout 上述 slime commit 就假定
获得全部功能。迁移到新服务器时，必须同步第 17 节列出的关键文件或先将这些改动正式提交。

依赖版本：

```text
SWE-Gym/SWE-Gym dataset revision:
  bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb

/data/xys/slime-coding-agent/deps/SWE-Gym:
  b681068ca20628c6987b7416cc4cf03f06b77ba5

/data/xys/slime-coding-agent/deps/SWE-Bench-Fork:
  242429c188fcfd06aad13fce9a54d450470bf0ac
```

## 3. 双服务器职责

```text
192.168.110.102 / master / xys
  - /home/xys/slime
  - /data/xys/slime-coding-agent
  - 2 x NVIDIA H200 NVL
  - Qwen3.5-35B-A3B HF 权重
  - Ray、SGLang、Megatron、adapter
  - rollout dump、日志、checkpoint

192.168.110.101 / cu01 / whz
  - Docker Engine 24.0.9
  - SWE-Gym task image
  - 临时 coding-agent task container
  - 临时 clean-eval/grader container
```

模型生成和训练都在 `.102` 的两张 H200 上。迁移到 `.101` 的只是 Agent 工具执行、代码修改
和 clean evaluation，不是 LLM token generation。

`.102` 通过 SSH 公钥建立 Unix-socket tunnel：

```text
.102 temporary docker-<PID>.sock
  -> SSH 22
  -> .101 /var/run/docker.sock
```

runtime 内使用：

```text
DOCKER_HOST=unix:///var/run/slime-remote-docker.sock
```

`.101` task container 通过物理网络访问：

```text
http://192.168.110.102:18001
```

## 4. 安全约束

文档、日志和命令历史中不要写入账号密码或私钥内容。服务器间只使用专用 SSH 公钥：

```text
/home/xys/.ssh/slime_docker_whz_192_168_110_101_ed25519
```

权限应为：

```bash
chmod 700 /home/xys/.ssh
chmod 600 /home/xys/.ssh/slime_docker_whz_192_168_110_101_ed25519
chmod 600 /home/xys/.ssh/known_hosts
```

验证 public-key only 登录：

```bash
ssh \
  -i /home/xys/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  -o BatchMode=yes \
  -o PasswordAuthentication=no \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/home/xys/.ssh/known_hosts \
  whz@192.168.110.101 \
  'hostname; id; docker version'
```

不要开放未加密 Docker TCP 2375，不要把 SSH 私钥挂载进 runtime 或 task image。

## 5. `.102` 前置资源

检查代码、模型、runtime image、GPU、磁盘和离线 Agent 资产：

```bash
cd /home/xys/slime

test -f /home/xys/ms-swift/model/Qwen/Qwen3.5-35B-A3B/config.json
test -f /data/xys/slime-coding-agent/assets/node-v22-linux-x64.tar.xz
test -f /data/xys/slime-coding-agent/assets/anthropic-ai-claude-code.tgz

docker image inspect slime-coding-agent-runtime:local
nvidia-smi
df -h /data
```

训练期间需要同时容纳：

```text
约 65 GiB 转换后的 Megatron reference checkpoint
约 121 GiB 临时 file-backed FP32 CPU master
约 65 GiB iteration checkpoint
Ray、Docker、日志和临时文件空间
```

建议 `/data` 在训练前至少有 300 GiB 可用空间。不要在宿主机安装 swap，也不要禁用 Ray
memory monitor。

## 6. `.101` Docker 前置检查

在 `.102` 执行：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b_swegym.sh preflight
```

成功时应看到：

```text
Sandbox node: whz@192.168.110.101
Remote Docker SSH tunnel: OK
```

也可直接检查 `.101`：

```bash
ssh \
  -i ~/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  -o BatchMode=yes \
  whz@192.168.110.101 \
  'docker version; docker info; df -h /var/lib/docker; docker ps'
```

`whz` 必须能够直接使用 `/var/run/docker.sock`。Docker 组权限等价于该节点上的 root 级
容器控制权限，专用 SSH key 必须限制用途和读取权限。

## 7. 固定 SWE-Gym 数据

当前固定数据文件：

```text
/data/xys/slime-coding-agent/datasets/swe-gym/SWE-Gym-train.parquet
rows: 2438
size: 43644473 bytes
sha256: 60569cea74bb281f7a5579467436a2bc1932c6e0c5f2f7fa0d084392abd9ad97
```

验证：

```bash
sha256sum \
  /data/xys/slime-coding-agent/datasets/swe-gym/SWE-Gym-train.parquet

cat /data/xys/slime-coding-agent/datasets/swe-gym/manifest.json
```

如需在新环境重新取得数据，应固定 dataset revision。以下命令只在 runtime 容器中使用
Python 依赖，不向宿主机 Python 安装包：

```bash
WORKSPACE=/data/xys/slime-coding-agent
DATASET_COMMIT=bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb

mkdir -p "${WORKSPACE}/datasets/swe-gym"

docker run --rm -i \
  --volume "${WORKSPACE}:/workspace" \
  slime-coding-agent-runtime:local \
  python3 - <<'PY'
from datasets import load_dataset

revision = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
dataset = load_dataset("SWE-Gym/SWE-Gym", revision=revision, split="train")
assert len(dataset) == 2438
dataset.to_parquet("/workspace/datasets/swe-gym/SWE-Gym-train.parquet")
PY
```

不同 `datasets`/`pyarrow` 版本重新编码 Parquet 时，文件级 SHA256 可能不同；此时必须同时
验证 revision、2438 行、字段内容和生成后的 manifest。当前服务器上继续使用已经固定且
校验通过的 Parquet，不要无理由重写它。

## 8. 生成 slime JSONL

转换工具：

```text
/data/xys/slime-coding-agent/deps/prepare_swegym_train.py
```

执行：

```bash
WORKSPACE=/data/xys/slime-coding-agent
DATASET_COMMIT=bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb

docker run --rm \
  --volume "${WORKSPACE}:/workspace" \
  slime-coding-agent-runtime:local \
  python3 /workspace/deps/prepare_swegym_train.py \
    --parquet /workspace/datasets/swe-gym/SWE-Gym-train.parquet \
    --output-dir /workspace/datasets/swe-gym \
    --candidate-count 32 \
    --image-namespace xingyaoww \
    --dataset-commit "${DATASET_COMMIT}"
```

验收：

```bash
wc -l /data/xys/slime-coding-agent/datasets/swe-gym/*.jsonl
```

预期：

```text
2438 swegym_train_full.jsonl
  32 swegym_train_candidates.jsonl
   1 swegym_train_first_task.jsonl
```

每行的 `prompt` 必须是 chat message list，不能是普通字符串：

```json
{"prompt":[{"role":"user","content":"..."}],"label":"...","metadata":{"dataset":"SWE-Gym/SWE-Gym","split":"train","remote_env_info":{"image":"...","workdir":"/testbed"}}}
```

字符串 prompt 会在 processor 路径触发：

```text
AssertionError: prompt must be a list when processor is not None
```

SWE-Gym image 名称规则为将 `instance_id` 中的 `__` 替换成 `_s_` 并转为小写。例如：

```text
iterative__dvc-2118
-> xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest
```

## 9. 固定 SWE-Gym grader 依赖

新环境中准备依赖：

```bash
WORKSPACE=/data/xys/slime-coding-agent
mkdir -p "${WORKSPACE}/deps"

git clone https://github.com/SWE-Gym/SWE-Gym.git \
  "${WORKSPACE}/deps/SWE-Gym"
git -C "${WORKSPACE}/deps/SWE-Gym" checkout \
  b681068ca20628c6987b7416cc4cf03f06b77ba5

git clone https://github.com/SWE-Gym/SWE-Bench-Fork.git \
  "${WORKSPACE}/deps/SWE-Bench-Fork"
git -C "${WORKSPACE}/deps/SWE-Bench-Fork" checkout \
  242429c188fcfd06aad13fce9a54d450470bf0ac
```

Ray runtime 固定使用：

```text
PYTHONPATH=/workspace/deps/swebench-bootstrap:/workspace/deps/SWE-Bench-Fork:/root/Megatron-LM:/root/slime
```

当前 `examples/coding_agent_rl/swe.py` 已兼容：

```text
swebench.harness.test_spec.test_spec
swebench.harness.test_spec
grader 参数 test_log_path / log_path
旧 grader 的实例目录日志结构
>>>>> Applied Patch (pred) 日志标记
```

验证单条任务可以构建 test spec 并进入 grader：

```bash
WORKSPACE=/data/xys/slime-coding-agent

docker run --rm \
  --volume /home/xys/slime:/root/slime \
  --volume "${WORKSPACE}:/workspace" \
  --workdir /root/slime \
  --env PYTHONPATH=/workspace/deps/swebench-bootstrap:/workspace/deps/SWE-Bench-Fork:/root/Megatron-LM:/root/slime \
  slime-coding-agent-runtime:local \
  python3 /workspace/deps/validate_slime_swegym.py \
    /workspace/datasets/swe-gym/swegym_train_first_task.jsonl
```

预期任务：

```text
instance_id: iterative__dvc-2118
repo: iterative/dvc
version: 0.41
workdir: /testbed
FAIL_TO_PASS tests: 2
PASS_TO_PASS tests: 13
```

## 10. 准备 `.101` 任务镜像

单步验收使用：

```text
xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest
image ID: sha256:9bed8d248fcf22bdc5a772243584d2b56e728a9214b8e65f841bb5fdcb56746a
digest: sha256:005bc979a0081fbf3fd51dcab2b9229092574f12da05f11bcbe73363a75f2bcd
size: 2797452878 bytes
architecture: amd64
workdir: /testbed
base git HEAD: dd7876e72f0317209144318c0639f8c1e8b00199
```

如 `.101` 能直接访问镜像源：

```bash
ssh \
  -i ~/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  whz@192.168.110.101 \
  'docker pull xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest'
```

如使用当前代理：

```bash
ssh \
  -i ~/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  whz@192.168.110.101 \
  'docker pull dockerproxy.net/xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest && \
   docker tag \
     dockerproxy.net/xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest \
     xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest'
```

验收：

```bash
ssh \
  -i ~/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  whz@192.168.110.101 \
  'docker image inspect xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest; \
   docker run --rm xingyaoww/sweb.eval.x86_64.iterative_s_dvc-2118:latest \
     sh -lc "test -d /testbed && cd /testbed && git rev-parse HEAD"'
```

使用 32 条候选或全量数据时，必须提前在 `.101` 准备对应 instance 的全部 task image；只拉
取 `iterative__dvc-2118` 不能支持其他 task。

## 11. 转换 Qwen3.5-35B-A3B

只需在首次部署或转换产物丢失时执行：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b_swegym.sh convert
```

输出：

```text
/data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist
```

验收：

```bash
test -f \
  /data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist/latest_checkpointed_iteration.txt
cat \
  /data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist/latest_checkpointed_iteration.txt
du -sh \
  /data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist
```

预期 marker 为 `release`，目录约 65 GiB。转换入口不会覆盖已存在的有效产物。

## 12. 运行真实 Rollout-only

先运行一条真实 task，验证模型、adapter、SSH tunnel、`.101` task/eval container 和官方
grader，不执行 backward：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b_swegym.sh rollout
```

定位最新运行：

```bash
latest="$(ls -dt \
  /data/xys/slime-coding-agent/runs/rollout_qwen35_35b_a3b_swegym_* \
  | head -1)"
echo "${latest}"

grep -nE \
  'swe\.swebench|coding_agent_rl|Finish rollout|Job .* succeeded|Traceback|Exception' \
  "${latest}/run.log"

test -s "${latest}/rollout_dumps/rollout_0.pt"
```

基础设施成功的判据是：

```text
patch_applied=True
grader exit_code=0
rollout_0.pt 存在
Ray Job succeeded
```

`reward=0` 只说明模型没有通过全部目标和回归测试，不能单独判定基础设施失败。

已验证 rollout-only：

```text
/data/xys/slime-coding-agent/runs/rollout_qwen35_35b_a3b_swegym_20260718_062837
Ray job: raysubmit_ygXPuzf94kjVXkrm
reward=0
exit_code=0
patch_applied=True
F2P=0/2
P2P=13/13
agent_exit_code=0
```

## 13. 运行真实单步训练

确认 rollout-only 成功后执行：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b_swegym.sh train
```

默认数据和训练规模：

```text
PROMPT_DATA=/workspace/datasets/swe-gym/swegym_train_first_task.jsonl
num_rollout=1
rollout_batch_size=1
n_samples_per_prompt=2
global_batch_size=2
num_steps_per_rollout=1
```

默认资源布局：

```text
Megatron actor: TP1 / PP1 / CP1 / EP2 / ETP1 / DP2
SGLang: TP2 / EP2
max_tokens_per_gpu=40000
log_probs_chunk_size=64
train_memory_margin_bytes=67108864
Ray object store=4 GiB
Ray memory threshold=0.98
NCCL_NVLS_ENABLE=0
```

完整单步约需 70 分钟。reference checkpoint、FP32 master 初始化、StatelessAdam 和 checkpoint
保存阶段可能数分钟到数十分钟没有新增日志。此时先检查进程状态和磁盘 I/O，不要立即重跑。

## 14. 训练监控

另开终端定位最新运行：

```bash
latest="$(ls -dt \
  /data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_swegym_* \
  | head -1)"
echo "${latest}"
```

查看关键事件：

```bash
grep -nE \
  'loading release distributed checkpoint|swe\.swebench|coding_agent_rl|actor train:|saving checkpoint|successfully saved checkpoint|Update weights:|Job .* succeeded|Traceback|Exception' \
  "${latest}/run.log" \
  | tail -200
```

查看 GPU：

```bash
watch -n 2 \
  'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'
```

查看临时 FP32 master：

```bash
du -h "${latest}/cpu_master"/* 2>/dev/null || true
```

两个文件逻辑大小各约 60 GiB。成功或失败退出时，脚本 trap 都应删除：

```text
cpu_master/rank_0.bin
cpu_master/rank_1.bin
```

检查本地 runtime：

```bash
docker ps --filter name=slime-coding-agent-main-remote-35b-a3b-swegym
```

检查 `.101` 临时容器：

```bash
ssh \
  -i ~/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  whz@192.168.110.101 \
  'docker ps'
```

## 15. 成功验收

训练结束后执行：

```bash
latest="$(ls -dt \
  /data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_swegym_* \
  | head -1)"

grep -nE \
  'swe\.swebench|coding_agent_rl|actor train: 100%|saving checkpoint|Update weights: 100%|Job .* succeeded' \
  "${latest}/run.log"

cat "${latest}/checkpoints/latest_checkpointed_iteration.txt"
du -sh "${latest}/checkpoints"
ls -lh "${latest}/checkpoints/iter_0000000"
test -s "${latest}/rollout_dumps/rollout_0.pt"
test ! -d "${latest}/cpu_master"
```

必须同时满足：

```text
两个 trajectory 都进入官方 grader
grader exit_code=0
rollout_0.pt 存在
actor dynamic microbatch 完成
optimizer step 完成
权重同步 128/128 完成
latest_checkpointed_iteration.txt 为 0
iteration 0 checkpoint 约 65 GiB
Ray Job succeeded
cpu_master 已清理
.102 无本次 runtime 容器残留
.101 docker ps 无 task/eval 容器残留
```

已验证成功运行：

```text
run root:
  /data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_swegym_20260718_063240

Ray job:
  raysubmit_98XKWkxd7Dtq9r4q

result:
  Ray Job succeeded
  checkpoint marker=0
  checkpoint size=65 GiB
  rollout_0.pt=1.4 MiB
  cpu_master cleaned
  .101 task/eval containers cleaned
```

本次两个 grader 结果：

```text
trajectory 1:
  agent_exit_code=0
  patch_applied=True
  F2P=0/2
  P2P=13/13
  reward=0

trajectory 2:
  agent_exit_code=1
  patch_applied=True
  F2P=2/2
  P2P=12/13
  reward=0
```

第二条解决了目标测试但破坏一个回归测试，所以标准二值 reward 为 0。两条 reward 相同导致
本组 GRPO advantage 为 0；这次运行证明训练机械链路成功，但不代表获得了有效正奖励学习
信号。正式 campaign 应使用多个 task 和更大的 prompt batch，降低整组同奖励概率。

## 16. 扩展到 32 条候选或 2438 条全量任务

不要直接修改已经通过验收的：

```text
run_qwen35_35b_a3b_swegym.sh
run_remote_docker_35b_a3b_swegym.sh
```

当前成功入口固定 `num_rollout=1`。仅把 `PROMPT_DATA` 指向 full JSONL 并不会自动跑完
2438 条任务，还必须设置 campaign 的 rollout/step 数、batch、保存周期和任务镜像计划。

建议先复制独立 campaign 脚本：

```bash
cd /home/xys/slime/examples/coding_agent_rl/local_docker

cp run_qwen35_35b_a3b_swegym.sh \
  run_qwen35_35b_a3b_swegym_campaign.sh
```

在 campaign 副本中至少单独配置：

```text
PROMPT_DATA=/workspace/datasets/swe-gym/swegym_train_candidates.jsonl
或 /workspace/datasets/swe-gym/swegym_train_full.jsonl

--num-rollout <计划步数>
--rollout-batch-size <prompt batch>
--n-samples-per-prompt <每题采样数>
--global-batch-size <与实际 sample 数一致>
--save-interval <合理周期，不建议全量训练每步保存 65 GiB>
```

然后创建独立 remote wrapper，令：

```text
SLIME_CODING_AGENT_RUN_SCRIPT=
  examples/coding_agent_rl/local_docker/run_qwen35_35b_a3b_swegym_campaign.sh
```

启动 campaign 前必须：

```text
1. 在 .101 准备本次 task 集合对应的所有 xingyaoww/sweb.eval 镜像。
2. 估算 Docker image、rollout、checkpoint 和 /data 空间。
3. 先用 2 到 4 个 task 验证 reward 分布不是整组恒定。
4. 降低 checkpoint 频率，避免每步额外写约 65 GiB。
5. 保留当前单步入口，作为回归和故障隔离基线。
```

## 17. 迁移到新环境时必须同步的本地文件

在未正式提交本地改动前，至少同步：

```text
examples/coding_agent_rl/swe.py
examples/coding_agent_rl/generate.py
slime/agent/harness/common.py
slime/agent/sandbox.py
examples/coding_agent_rl/local_docker/run_remote_docker.sh
examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b_swegym.sh
examples/coding_agent_rl/local_docker/run_qwen35_35b_a3b_swegym.sh
examples/coding_agent_rl/local_docker/qwen35_35b_cpu_offload.py
/data/xys/slime-coding-agent/deps/prepare_swegym_train.py
/data/xys/slime-coding-agent/deps/validate_slime_swegym.py
```

当前关键 SHA256：

```text
847538aeb551b1dc4dbdfb76dc490256a4e8893265c2fb613240a6f4a9324269  examples/coding_agent_rl/swe.py
e72c176f04565585c91f6c62db1d5961df962cd81f9f73fcedeefb04ed7dd5bd  examples/coding_agent_rl/generate.py
f44fd4b19f45fa0bcc187627905f40c62718713018e44ccbbb626afad11bbff0  slime/agent/harness/common.py
bdb8f3f84542252077f4667f9f6be9df8e5416cbc1c3662969214648ae13092a  slime/agent/sandbox.py
51a4842c6f2c796e7ad4b834aba90ad61af0b5f0a9838557f4fcfe360fde5647  run_remote_docker.sh
e8304a491f2bec2ffde9d9698fe8ee8235d0f02d9e2073905255363650b65171  run_remote_docker_35b_a3b_swegym.sh
67bba795ce6c7621286d0b8e5baef9957da938daa017afafc324398c3f116803  run_qwen35_35b_a3b_swegym.sh
28fd238617c9ac2a9ef8b98ce8e454ffbe6b392b935c32df80cbbb7bda0e100c  qwen35_35b_cpu_offload.py
6433dec21492fc37e240ce8da7b3b0ca96847a14543951e9463bcf86d0e9c653  prepare_swegym_train.py
ec24a3ba07de7a513efda4c88cde1093b3956aa8f3ac9da66dc1ca74b2a56a61  validate_slime_swegym.py
```

这些校验值用于识别环境漂移，不应替代把改动提交到版本控制。

## 18. 常见问题

### 18.1 prompt 类型错误

```text
AssertionError: prompt must be a list when processor is not None
```

原因：JSONL 的 `prompt` 是字符串。重新用 `prepare_swegym_train.py` 生成 chat list。

### 18.2 `.101` 缺少任务镜像

```text
Remote task image is missing
```

在 `.101` 拉取并 retag 对应 `instance_id` 的 `xingyaoww/sweb.eval` 镜像。

### 18.3 grader reward 为 0

先区分模型失败和基础设施失败：

```text
exit_code=0 + patch_applied=True + F2P/P2P 有结果
  -> grader 正常，模型解题未满足全部测试

镜像启动失败、patch_applied=False、grader traceback
  -> 基础设施或数据问题
```

### 18.4 Agent 达到 40k context

日志可能出现：

```text
prompt exceeds max_context_tokens
agent_exit_code=1
```

只要 patch 已提取并且 grader `exit_code=0`，该 trajectory 仍能被评测。正式训练应监控此比例，
必要时改善任务提示、Agent 工具策略或 context 配置，而不是把它误判为 Docker 断连。

### 18.5 `TorchMemorySaver::malloc return OOM`

该底层日志可能表示 memory saver 拒绝大块额外分配并执行回退。必须看后续是否完成所有
microbatch；如果出现 Python/CUDA traceback 或 Ray actor death 才是最终 OOM。本次成功运行
曾打印该信息，随后仍完成 `5/5` microbatch、optimizer 和 checkpoint。

### 18.6 optimizer 长时间没有日志

StatelessAdam 会扫描约 121 GiB file-backed FP32 master。两个 actor 处于
`folio_wait_bit_common`、CPU 时间持续增长且 Ray job 为 RUNNING，通常是磁盘 I/O，不是 NCCL
死锁。本次 optimizer 阶段约 31 分钟。

### 18.7 失败日志

失败时入口自动归档：

```text
${RUN_ROOT}/ray_logs
```

优先检查：

```bash
tail -n 300 "${latest}/run.log"
find "${latest}/ray_logs" -maxdepth 2 -type f -print 2>/dev/null
```

不要修改宿主机 swap，不要设置 `RAY_memory_monitor_refresh_ms=0`，不要覆盖已经成功的 smoke
和单步 SWE-Gym 入口。

## 19. 回退路径

回到 35B calculator smoke：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh rollout
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh train
```

回到原 remote 4B：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh rollout
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh train
```

回到原单机流程：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_host.sh rollout
bash examples/coding_agent_rl/local_docker/run_host.sh train
```

SWE-Gym 入口和本文档均为独立新增内容，不要求删除或改写上述已验证流程。
