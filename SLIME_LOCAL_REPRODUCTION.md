# slime Coding Agent 本机 Docker 可复现流程

## 1. 复现范围

本文复现以下已验证闭环：

```text
真实 DockerSandbox
  -> Claude Code 安装
  -> Qwen3.5-4B HF 权重转换
  -> 单条 coding-agent rollout
  -> 全新容器 clean eval
  -> 单步 GRPO
  -> torch_dist checkpoint
```

本文针对 2026-07-17 的本机环境。命令默认使用物理 GPU 1，并依赖当前
`/home/xys/slime` 工作树中的未提交 DockerSandbox 修改。不要只重新检出
`fb42ae45`：该提交本身不包含这些代码。

## 2. 已验证环境

- 仓库：`/home/xys/slime`
- 基准提交：`fb42ae45`
- GPU：H200 NVL 141GB；脚本固定 `--gpus device=1`
- HF 模型：`/home/xys/models/Qwen3.5-4B`
- 大容量工作目录：`/data/xys/slime-coding-agent`
- Node.js：`22.20.0` Linux x64
- Claude Code：`2.1.212`
- runtime image：`slime-coding-agent-runtime:local`
- task image：`slime-coding-agent-swe-smoke:local`

建议至少预留：

- Docker 镜像空间约 22GB。
- 转换模型约 8GB。
- 单次训练 checkpoint 约 55GB。
- 额外日志、rollout dump 和构建缓存空间。

## 3. 前置检查

```bash
cd /home/xys/slime

git rev-parse --short HEAD
git status --short
test -f /home/xys/models/Qwen3.5-4B/config.json
docker version
nvidia-smi
df -h /home/xys /data
```

预期：

- `git rev-parse` 以 `fb42ae45` 为基准。
- `git status` 能看到本次未提交的修改和 `local_docker/` 新文件。
- Docker 能使用 NVIDIA runtime。
- 物理 GPU 1 可用。

如果要换 GPU，需要修改
`examples/coding_agent_rl/local_docker/run_host.sh` 中的
`--gpus device=1`。容器内仍应使用 `CUDA_VISIBLE_DEVICES=0`。

## 4. 准备目录和 harness 资产

```bash
mkdir -p \
  /data/xys/slime-coding-agent/assets \
  /data/xys/slime-coding-agent/models \
  /data/xys/slime-coding-agent/runs

curl -fL --retry 3 \
  -o /data/xys/slime-coding-agent/assets/node-v22-linux-x64.tar.xz \
  https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz

npm pack @anthropic-ai/claude-code@2.1.212 \
  --pack-destination /data/xys/slime-coding-agent/assets

ln -sfn anthropic-ai-claude-code-2.1.212.tgz \
  /data/xys/slime-coding-agent/assets/anthropic-ai-claude-code.tgz
```

校验：

```bash
sha256sum \
  /data/xys/slime-coding-agent/assets/node-v22-linux-x64.tar.xz \
  /data/xys/slime-coding-agent/assets/anthropic-ai-claude-code-2.1.212.tgz
```

预期输出：

```text
00bbd05e306ea68b6e13e17360d0e2f680b493ef95f2fea1c4296ff7437530bc  node-v22-linux-x64.tar.xz
2162841dd793d21671eccb7fe76fe9c3da6816adf447ba3890a4871b7e5f4e69  anthropic-ai-claude-code-2.1.212.tgz
```

## 5. 构建镜像

```bash
cd /home/xys/slime

docker build \
  -t slime-coding-agent-swe-smoke:local \
  -f examples/coding_agent_rl/local_docker/Dockerfile.swe_smoke \
  examples/coding_agent_rl/local_docker

docker build \
  -t slime-coding-agent-runtime:local \
  -f examples/coding_agent_rl/local_docker/Dockerfile.runtime \
  .
```

2026-07-17 已验证的本地镜像身份：

```text
runtime: sha256:b327465cfd4b73a48c7f1dc0e1e48be9c8cef8dd7d4c2820dfead5b915c95593
task:    sha256:172e4c1e1e87b2740c4c72fa01a01bd859204f758aec930cd45d88eee2f7a402
```

`Dockerfile.runtime` 当前以 `slimerl/slime:latest` 为基础。跨时间或跨机器做严格
复现时，应先把该基础镜像改成组织内可访问的固定 digest；否则重新构建可能得到
不同依赖版本。上面的本地 runtime digest 不能替代未推送的基础镜像引用。

## 6. 分层验证

### 6.1 Agent 测试

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh test
```

已验证结果：

```text
65 passed, 1 skipped
```

### 6.2 DockerSandbox 集成 smoke

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh sandbox-smoke
```

该步骤不加载模型，验证：

- 创建和清理 sibling container。
- exec/read/write。
- 以 agent 用户写工作目录。
- 运行 pytest 后只提取源码 diff，不包含 pyc/pytest 缓存。
- 在全新评测容器中应用 diff 并得到 1 分。

### 6.3 Claude Code 安装 smoke

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh claude-smoke
```

预期输出包含：

```text
2.1.212 (Claude Code)
```

此步骤只验证 harness 安装，不请求 Anthropic 服务。

## 7. 转换 Qwen3.5-4B

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh convert
```

脚本具有存在性短路：如果目标目录已有
`latest_checkpointed_iteration.txt`，会直接退出而不覆盖。

验收：

```bash
du -sh /data/xys/slime-coding-agent/models/Qwen3.5-4B_torch_dist
cat /data/xys/slime-coding-agent/models/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt
find /data/xys/slime-coding-agent/models/Qwen3.5-4B_torch_dist \
  -maxdepth 2 -type f -printf '%s %p\n' | sort -nr | head
```

已验证结果：约 `7.9GB`，iteration 为 `release`。

## 8. Rollout-only

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh rollout
```

脚本会打印本次 `RUN_ROOT`，例如：

```text
/data/xys/slime-coding-agent/runs/rollout_20260717_090950
```

找到最新日志并验收：

```bash
latest_rollout="$(find /data/xys/slime-coding-agent/runs \
  -maxdepth 1 -type d -name 'rollout_*' | sort | tail -1)"

rg -n \
  'reward=1.00|applied=True|agent_exit_code=0|Job .* succeeded|response_len' \
  "${latest_rollout}/run.log"

test -f "${latest_rollout}/rollout_dumps/rollout_0.pt"
```

成功门槛必须同时满足：

```text
reward=1.00
applied=True
agent_exit_code=0
Ray Job succeeded
```

已验证运行的 response length 为 418。模型采样具有随机性，复跑时长度不必完全
相同；上述四个成功条件才是验收标准。

## 9. 单步 GRPO

只有 rollout 成功后再执行：

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh train
```

当前 train 参数：

```text
n_samples_per_prompt=2
global_batch_size=2
num_rollout=1
num_steps_per_rollout=1
lr=1e-6
save_interval=1
```

找到最新训练目录并验收：

```bash
latest_train="$(find /data/xys/slime-coding-agent/runs \
  -maxdepth 1 -type d -name 'train_*' | sort | tail -1)"

rg -n \
  'reward=1.00|step 0:|successfully saved checkpoint|Timer save_model end|Job .* succeeded' \
  "${latest_train}/run.log"

cat "${latest_train}/checkpoints/latest_checkpointed_iteration.txt"
find "${latest_train}/checkpoints" -maxdepth 2 -type f \
  -printf '%p %s bytes\n' | sort
du -sh "${latest_train}/checkpoints"
```

已验证输出：

- 两条轨迹 clean eval 均为 1。
- 日志出现 `step 0`，`global_batch_size=2`，`lr=1e-6`。
- 日志出现 `successfully saved checkpoint from iteration 0`。
- `latest_checkpointed_iteration.txt` 内容为 `0`。
- checkpoint 约 55GB。
- Ray job succeeded。

## 10. 结果位置

已验证的最终运行：

```text
rollout log:
  /data/xys/slime-coding-agent/runs/rollout_20260717_090950/run.log

train log:
  /data/xys/slime-coding-agent/runs/train_20260717_091310/run.log

checkpoint:
  /data/xys/slime-coding-agent/runs/train_20260717_091310/checkpoints
```

## 11. 常见失败检查

### Ray job agent 无法连接

检查以下设置仍在脚本中：

- `MASTER_ADDR` 使用容器网卡地址，不是强制 loopback。
- HTTP(S) proxy 在启动 Ray 前清空。
- `no_proxy` 包含 localhost、容器节点地址和主容器名。
- Ray 使用 `--num-cpus 8`。
- 主容器使用 `--ulimit nofile=1048576:1048576`。

失败运行会把 Ray 日志归档到 `${RUN_ROOT}/ray_logs`。

### Prompt 类型错误

Qwen3.5-4B 会创建 processor；`smoke.jsonl` 中 `prompt` 必须是消息列表，不能是
纯字符串。

### 请求在生成前被拒绝

Claude Code 的系统提示和工具 schema 约 17.6K tokens。保持
`MAX_CONTEXT_LEN>=32768`；16K 不够。

### Clean eval applied=False

检查：

- patch 文件由 root 写入后已 `chown agent`。
- diff 没有 `__pycache__`、`*.pyc`、`.pytest_cache`。
- patch fallback 使用 check/dry-run 后再 apply，失败尝试不会污染工作树。

### npm EACCES 或 tarball 不可读

稳定 tarball 名是符号链接。DockerSandbox 必须以 `docker cp --follow-link` 复制
实际文件，并确保容器内 `/tmp/harness-cli.tgz` 权限为 `0644`。

## 12. 结果解释与安全边界

两条 smoke 轨迹都得到 1 分，GRPO 归一化后 advantage 为 0，因此本次运行的
loss 和 grad norm 为 0。它验证了工程通路，不验证有效学习。要验证非零更新，
应改用能稳定产生奖励差异的多样本任务。

主容器挂载了 `/var/run/docker.sock`。这等价于给予主容器宿主机 Docker/root 级
权限。不要在此配置下运行不可信的模型生成代码、任务仓库、任务镜像或 harness
资产。

