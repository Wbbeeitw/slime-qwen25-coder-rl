# slime Qwen3.5-35B-A3B 双 H200 Remote Docker 复现流程

## 1. 范围与机器分工

本流程建立在已经跑通的 Qwen3.5-4B remote Docker 流程之上，并保持原流程可回退：

```text
192.168.110.102
  - /home/xys/slime
  - Ray / Megatron / SGLang
  - Qwen3.5-35B-A3B rollout 和训练
  - NVIDIA H200 NVL GPU 0、1

192.168.110.101
  - 只运行 coding-agent DockerSandbox 和 clean eval 容器
  - 由 .102 通过 SSH Unix-socket tunnel 访问 Docker daemon
```

不要修改或删除以下原 4B 入口：

```text
/home/xys/slime/SLIME_LOCAL_REPRODUCTION.md
examples/coding_agent_rl/local_docker/run_host.sh
examples/coding_agent_rl/local_docker/run_remote_docker.sh
```

35B 专用入口：

```text
examples/coding_agent_rl/local_docker/convert_qwen35_35b_a3b.sh
examples/coding_agent_rl/local_docker/run_qwen35_35b_a3b.sh
examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh
examples/coding_agent_rl/local_docker/qwen35_35b_cpu_offload.py
```

## 2. 已验证环境

```text
仓库：/home/xys/slime
HF 模型：/home/xys/ms-swift/model/Qwen/Qwen3.5-35B-A3B
转换模型：/data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist
工作目录：/data/xys/slime-coding-agent
runtime image：slime-coding-agent-runtime:local
task image：slime-coding-agent-swe-smoke:local
GPU：2 x NVIDIA H200 NVL，每卡约 141 GiB
sandbox：whz@192.168.110.101 的 Docker 24.0.9
```

两张 H200 之间没有 NVLink，运行时保留：

```bash
export NCCL_NVLS_ENABLE=0
```

模型转换产物约 65 GiB，训练还需要约 121 GiB 临时 file-backed FP32 master，
以及约 65 GiB 级别的训练 checkpoint。建议 `/data` 至少预留 300 GiB。

## 3. 前置检查

```bash
cd /home/xys/slime

bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh preflight
test -f /home/xys/ms-swift/model/Qwen/Qwen3.5-35B-A3B/config.json
docker image inspect slime-coding-agent-runtime:local
nvidia-smi
df -h /data
```

`.101` 的 task image 如果尚未同步，执行一次：

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh build
```

## 4. 转换模型

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh convert
```

脚本发现目标目录已有 `latest_checkpointed_iteration.txt` 时会直接退出，不覆盖已转换权重。

验收：

```bash
du -sh /data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist
cat /data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist/latest_checkpointed_iteration.txt
```

已验证 iteration 为 `release`，目录约 65 GiB。

## 5. 并行布局

Megatron actor：

```text
TP=1, PP=1, CP=1, EP=2, ETP=1, DP=2
```

SGLang rollout：

```text
TP=2, EP=2, moe_dense_tp_size=1
mem_fraction_static=0.55
max_running_requests=2
```

不要把 actor 改回 `TP=2, EP=1, ETP=2, DP=1`。该布局能加载 checkpoint，
但 actor-to-SGLang 权重同步会出现 expert shard 维度不一致：

```text
RuntimeError: start (0) + length (512) exceeds dimension size (256)
```

actor `TP1/EP2` 与 SGLang `TP2/EP2` 已验证可以同步。

## 6. Rollout-only

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh rollout
```

已验证运行：

```text
/data/xys/slime-coding-agent/runs/rollout_qwen35_35b_a3b_20260717_124448
reward=1.00
applied=True
agent_exit_code=0
Ray Job succeeded
```

验收最新运行：

```bash
latest="$(ls -dt /data/xys/slime-coding-agent/runs/rollout_qwen35_35b_a3b_* | head -1)"
grep -nE 'reward=1.00|applied=True|agent_exit_code=0|Job .* succeeded' "${latest}/run.log"
test -f "${latest}/rollout_dumps/rollout_0.pt"
```

## 7. 单步训练

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh train
```

关键参数：

```text
n_samples_per_prompt=2
global_batch_size=2
num_rollout=1
num_steps_per_rollout=1
lr=1e-6
weight_decay=0.1
max_tokens_per_gpu=40000
log_probs_chunk_size=64
train_memory_margin_bytes=67108864
Ray object store=4 GiB
Ray memory monitor threshold=0.98
```

当前 smoke 数据是 `examples/coding_agent_rl/local_docker/smoke.jsonl` 中的本地
calculator 修复场景：要求把 `add` 的减法改为加法，再在 clean Docker 中跑 pytest。
它用于验证机械闭环，不是正式训练数据集。本次两条 trajectory 都得到 reward 1，
因此 GRPO advantage、policy loss 和 grad norm 均为 0；这不代表优化器没有执行。

`max_tokens_per_gpu=40000` 不能随意降回 32768。两条 trajectory 会形成三个约 18k
token 的 segment；32768 会产生三个 dynamic-batch bin，而 DP=2 要求偶数 bin，报错：

```text
could only produce 3 mbs; need 4
```

40000 可以把其中两个 segment 合并，最终形成两个 DP bin。

## 8. 35B 内存策略

### 8.1 FP32 master 与 BF16 gradient staging

`qwen35_35b_cpu_offload.py` 只由 35B 脚本加载，不影响 4B。它将每个 rank 的
FP32 optimizer master 映射到：

```text
${RUN_ROOT}/cpu_master/rank_0.bin
${RUN_ROOT}/cpu_master/rank_1.bin
```

使用 `torch.from_file(..., shared=True, dtype=float32)`，不安装宿主机 swap，也不需要
root。CPU staging gradient 和 Megatron main grad 使用 BF16：

```text
--grad-reduce-in-bf16
--main-grads-dtype bf16
```

精度取舍是 gradient reduce/staging 为 BF16；FP32 master 仍保留 `1e-6` 小学习率的
累计更新，避免每步直接写回 BF16 时被舍入掉。

指定 `--ref-load` 时 master 初始化延迟到 distributed checkpoint 加载，跳过从随机
初始化模型复制一次完整 FP32 master。脚本退出 trap 会删除本次约 121 GiB 临时文件。

每次 optimizer step 完成后，35B 专用 hook 会解除 CPU master 上的
`decoupled_grad` 引用并清空 `cpu_copy_map_grad`，释放两张 EP rank 合计约 60 GiB 的
BF16 pinned staging buffer。后续训练 step 会按需重新创建 staging buffer，参数更新和
FP32 master 数值不受影响。最终成功运行中，step 后 `host_used` 从未清理版本的约
194 GiB 降至约 127 GiB，使 checkpoint 保存可以继续保留 Ray 98% 内存保护。

### 8.2 StatelessAdam 语义

训练使用：

```text
--use-stateless-adam
--optimizer-cpu-offload
--use-precision-aware-optimizer
--no-save-optim
```

StatelessAdam 每一步仍更新所有参数，但不保存跨 step 的 `exp_avg`/`exp_avg_sq`。
checkpoint 不保存 optimizer state。它会完整扫描 file-backed master，因此本机 `/data`
速度下单步 optimizer 可能需要约 40 分钟；长时间没有新日志不等于卡死，可用
`vmstat 1 3` 确认持续 I/O。

### 8.3 Loss 峰值

35B 的完整 full-vocabulary logits 保持 BF16，不在模型 forward 结束时整体转成 FP32。
temperature scaling 和 softmax 只在每个 64-token chunk 内转成 FP32。专用 hook 做三项处理：

1. 覆盖 `Float16Module.forward` 的默认输出行为，使完整 logits 保持 BF16，避免一次约
   34 GiB 的 full-vocabulary BF16 -> FP32 dtype expansion。
2. log-prob 按 64 token 分块，每块在 FP32 中执行 `rollout_temperature` 缩放和 softmax。
3. forward 不保留每块 full-vocab softmax，backward 时重算，减少跨 backward 保留的
   full-vocabulary 中间张量。

第三项已经用普通分布、top-p mask、entropy gradient 的 CUDA 小张量测试与 PyTorch
参考实现对比通过。64-token scratch 约 61 MiB；配合 64 MiB TorchMemorySaver margin，
可覆盖随机 rollout 导致 full logits 大小变化的情况。最终成功运行中，`nvidia-smi`
监控采样峰值约为 GPU0 130691 MiB、GPU1 106583 MiB，已完成 backward、optimizer
step 和 checkpoint 保存。

### 8.4 Ray memory monitor

Ray memory monitor 保持启用，object store 限制为 4 GiB。保存 checkpoint 时两个 actor
会短暂同时唤醒；95% 默认阈值实测在 240.65/251.41 GiB 时仅超出约 24 MiB并误杀
save worker。仅将阈值提高到 98% 仍不够：未释放 BF16 staging gradient 时，保存阶段
在 248.41/251.41 GiB 再次超出 98% 阈值约 26 MiB。当前实现会在 optimizer step 后
释放 staging buffer，并继续使用 98% 阈值；最终完整训练与保存已通过：

```bash
RAY_MEMORY_USAGE_THRESHOLD=0.98 \
  bash examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh train
```

不要设置 `RAY_memory_monitor_refresh_ms=0`；那会完全禁用保护。

## 9. 训练验收

```bash
latest="$(ls -dt /data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_* | head -1)"

grep -nE \
  'reward=1.00|step 0:|grad_norm|successfully saved checkpoint|Timer save_model end|Job .* succeeded' \
  "${latest}/run.log"

cat "${latest}/checkpoints/latest_checkpointed_iteration.txt"
du -sh "${latest}/checkpoints"
test ! -d "${latest}/cpu_master"
```

完整成功必须同时满足：

```text
两条 remote rollout 完成
actor forward/backward 完成
optimizer step 完成并打印 step 0 指标
iteration 0 checkpoint 保存完成
Ray Job succeeded
退出后 cpu_master 已清理
```

最终完整成功运行：

```text
/data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_20260718_031727
Ray job: raysubmit_gyexLHDqA4QXQHLf
checkpoint marker: 0
checkpoint size: 65 GiB
save_model_time: 374.2s
Ray Job succeeded
cpu_master cleaned
```

最终实测训练指标：

```text
train/loss=0.0
train/pg_loss=0.0
train/entropy_loss=0.10291240073704605
train/grad_norm=0.0
train/lr=1e-6
actor_train_time=2048.1s
actor_train_tok_per_s=27.085979450224347
```

## 10. 回退到原 4B 流程

remote Docker 4B：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh rollout
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh train
```

原单机 4B：

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_host.sh rollout
bash examples/coding_agent_rl/local_docker/run_host.sh train
```

35B 文件均为新增专用入口；原 4B 复现文档和 `run_host.sh` 未修改。
