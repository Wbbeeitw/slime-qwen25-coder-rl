# slime Qwen3.5-35B-A3B 双 H200 Remote Docker 完整调试过程

## 1. 文档目的与边界

本文记录 2026-07-17 至 2026-07-18 将已经跑通的 slime remote 流程从
Qwen3.5-4B 扩展到 Qwen3.5-35B-A3B 时的完整调试过程，包括失败现象、根因、方案取舍和
最终验收证据。它是故障档案，不是日常操作手册；可直接执行的步骤见
`SLIME_REMOTE_DOCKER_35B_A3B_REPRODUCTION.md`。

原有流程保留不动：

```text
/home/xys/slime/SLIME_LOCAL_REPRODUCTION.md
/home/xys/slime/examples/coding_agent_rl/local_docker/run_host.sh
/home/xys/slime/examples/coding_agent_rl/local_docker/run_remote_docker.sh
```

本文不记录账号密码。日常连接使用本地 SSH 公钥配置：

```powershell
ssh -F E:\Data\llm\.codex-ssh-xys-110-102.conf xys-110-102
```

## 2. 最终机器拓扑

```text
192.168.110.102（训练机）
  /home/xys/slime
  Ray + Megatron actor + SGLang rollout model
  Qwen3.5-35B-A3B
  2 x NVIDIA H200 NVL，每卡约 141 GiB
          |
          | SSH Unix-socket tunnel / remote DOCKER_HOST
          v
192.168.110.101（sandbox 机）
  Docker 24.0.9
  coding-agent DockerSandbox
  每条 trajectory 的 clean eval 容器
```

这里需要区分两个容易混淆的概念：模型生成 rollout 的 SGLang 仍运行在 `.102` 的两张
H200 上；迁移到 `.101` 的是 rollout 过程中 agent 使用的 DockerSandbox 和测试容器。
因此 `.101` 不参与模型训练或推理，也不需要复制模型权重。

## 3. 初始评估

### 3.1 模型与资源

```text
HF checkpoint：/home/xys/ms-swift/model/Qwen/Qwen3.5-35B-A3B
模型类型：MoE，256 experts，top-k=8
转换后权重：约 65 GiB
GPU：2 x H200 NVL，总显存约 282 GiB，但显存不能视为一个无边界共享池
主机可见内存：约 251.41 GiB
```

35B-A3B 虽然每 token 只激活部分参数，训练时仍必须保存、同步和更新完整参数。仅依据
“3B active”估算训练内存会严重偏低。两卡之间没有可用 NVLink 路径，因此所有运行固定：

```bash
export NCCL_NVLS_ENABLE=0
```

### 3.2 预判的主要峰值

1. actor BF16 参数与训练中间张量占用 GPU 显存。
2. full-vocabulary logits 的词表大小为 248320，长 trajectory 下 BF16 logits 已很大；整体
   转成 FP32 会额外申请约 34 GiB。
3. 常规 Adam 的 FP32 master、`exp_avg` 和 `exp_avg_sq` 无法在现有 CPU/GPU 内存预算中
   同时常驻。
4. actor 和 SGLang colocate，通过 sleep/wake 复用 GPU，但 wake、权重同步、loss backward
   和 checkpoint 保存仍会形成不同阶段的峰值。
5. Ray 默认内存保护会在主机内存高于 95% 时主动杀 worker，不能把 Ray worker 消失简单
   判断为 CUDA 或网络故障。

因此调试顺序固定为：先转换权重，再验证 rollout-only 和远端 sandbox，最后才进入包含
forward、backward、optimizer 和 checkpoint 的单步训练。

## 4. 建立 35B 专用入口

为了不破坏已跑通的 4B 流程，新增四个 35B 专用文件：

```text
examples/coding_agent_rl/local_docker/convert_qwen35_35b_a3b.sh
examples/coding_agent_rl/local_docker/run_qwen35_35b_a3b.sh
examples/coding_agent_rl/local_docker/run_remote_docker_35b_a3b.sh
examples/coding_agent_rl/local_docker/qwen35_35b_cpu_offload.py
```

外层 wrapper 复用原 `run_remote_docker.sh` 的 SSH tunnel、DockerSandbox 和容器编排逻辑，
只覆盖模型路径、两卡设备、主容器名及 35B 启动脚本。这样 35B 的内存 hook 和实验参数
不会泄漏到原 4B 入口。

## 5. 模型转换与 rollout-only 验证

### 5.1 Hugging Face 权重转换

先将 HF checkpoint 转为 Megatron 可加载的 torch distributed checkpoint：

```text
输入：/home/xys/ms-swift/model/Qwen/Qwen3.5-35B-A3B
输出：/data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist
marker：latest_checkpointed_iteration.txt = release
大小：约 65 GiB
```

转换脚本具备幂等保护：目标目录已有 marker 时退出，不覆盖已验证权重。

### 5.2 先隔离验证生成链路

rollout-only 成功运行：

```text
/data/xys/slime-coding-agent/runs/rollout_qwen35_35b_a3b_20260717_124448
reward=1.00
applied=True
agent_exit_code=0
Ray Job succeeded
```

这一步证明以下链路已经成立：SGLang 在 `.102` 双 H200 加载 35B 模型、agent 能通过
远端 Docker daemon 在 `.101` 创建 sandbox、补丁能应用、clean eval 能完成并回传 reward。
后续失败因此集中在 actor 训练内存、并行布局和保存阶段，而不是 remote Docker 基础链路。

## 6. 17 次训练运行时间线

所有运行目录均保留在：

```text
/data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_YYYYMMDD_HHMMSS
```

| 序号 | 运行时间戳 | 结果与直接现象 | 定位结论 / 下一步 |
|---:|---|---|---|
| 1 | `20260717_124935` | `TorchMemorySaver` 与 CUDA `expandable_segments` 不兼容 | 移除该 allocator 配置，先恢复可用的 sleep/wake 内存路径 |
| 2 | `20260717_125331` | 初始化阶段 CUDA OOM，单次尝试申请 `60.00 GiB` | 默认训练状态不能直接常驻 GPU，转向 optimizer CPU offload |
| 3 | `20260717_125943` | Ray 在 `244.72/251.41 GiB`、95% 阈值杀 actor | 主机内存成为第一约束，需要减少 optimizer 常驻状态 |
| 4 | `20260717_131305` | Ray 在 `243.26/251.41 GiB`、95% 阈值再次杀 actor | 不是偶发进程故障，FP32 master/梯度副本仍超预算 |
| 5 | `20260717_131904` | Ray 在 `243.47/251.41 GiB`、95% 阈值再次杀 actor | 确认仅重试无效，继续拆解 CPU optimizer 内存 |
| 6 | `20260717_132626` | `ActorUnavailable`、RPC socket closed，另一 rank 随后 broken pipe | worker 被终止后的级联症状，不把 RPC/broken pipe 当根因 |
| 7 | `20260717_134131` | dynamic batch 只生成 3 个 microbatches，DP=2 需要 4 个 | `max_tokens_per_gpu=32768` 的分箱数不满足 DP 对齐，改为 40000 |
| 8 | `20260717_143006` | loss 阶段 CUDA OOM，尝试申请 `33.98 GiB` | full logits 被整体从 BF16 扩展为 FP32；必须保留 BF16、分块转 FP32 |
| 9 | `20260717_145845` | actor `TP2/EP1/ETP2` 能加载，但同步时报 expert shard mismatch | actor 与 SGLang expert 分片语义不一致，actor 改为 `TP1/EP2/ETP1` |
| 10 | `20260717_152905` | softmax scratch CUDA OOM，尝试申请 `486 MiB`，chunk=512 | 分块方向正确，但 chunk 太大；继续缩小 scratch |
| 11 | `20260717_155809` | chunk=128 时仍因约 `122 MiB` 申请 OOM | 显存余量小于 128-token scratch，降至 64 并预留 saver margin |
| 12 | `20260717_163303` | 再次出现 `486 MiB` CUDA OOM | 路径仍有 512-token 等效峰值，统一 hook 与实际 loss 分块参数 |
| 13 | `20260717_171023` | forward/backward/optimizer 成功，checkpoint 保存时被 Ray 95% 杀死 | 训练计算已跑通；保存并发唤醒使 host memory 短时过阈值 |
| 14 | `20260717_182342` | loss/backward 路径再次出现 `486 MiB` CUDA OOM | 继续消除保存至 backward 的 full-vocab softmax，改为 backward 重算 |
| 15 | `20260718_014534` | 再次尝试分配 `33.98 GiB` full FP32 logits | 找到 `Float16Module.forward` 默认输出转换，显式保持完整 logits 为 BF16 |
| 16 | `20260718_021502` | 训练 step 成功；保存时 `248.41/251.41 GiB`，超过 Ray 98% 约 26 MiB | 单纯提高阈值仍不够，step 后必须主动释放 BF16 pinned staging gradient |
| 17 | `20260718_031727` | 两条 rollout、训练、optimizer、iteration 0 保存、Ray 退出全部成功 | 最终方案通过完整验收 |

时间线中的 CUDA 分配值是失败请求大小，不是该阶段总显存；Ray 数值是整机已用内存与
总内存。RPC socket closed、broken pipe 和 `ActorUnavailable` 通常是首个 worker 已因 OOM
或 Ray memory monitor 被杀后的次生错误，应向前查找日志中的第一个异常。

## 7. 分组根因分析

### 7.1 GPU 显存：full-vocabulary loss 峰值

最危险的不是模型权重本身，而是长序列乘以 248320 词表产生的 logits。早期实现让
`Float16Module.forward` 把完整 BF16 logits 转为 FP32，直接增加约 33.98 GiB 峰值。
之后即使按 token 分块，如果 forward 为 backward 保存每块 full-vocab softmax，仍会因
512-token 或 128-token scratch 触发 486 MiB / 122 MiB 的临界 OOM。

最终实现：

```text
完整 logits：BF16
loss chunk：64 tokens
temperature scaling / softmax：只在当前 chunk 转 FP32
backward：重算 chunk softmax，不跨 forward/backward 保存完整 softmax
TorchMemorySaver margin：64 MiB
```

最终 `nvidia-smi` 采样峰值约 GPU0 130691 MiB、GPU1 106583 MiB，并完成 backward、
optimizer 和保存。没有选择全 BF16 softmax，因为概率与 entropy 梯度更需要 FP32 数值
稳定性；没有继续使用大 chunk，因为它只提升少量吞吐，却跨过了实际显存余量。

### 7.2 CPU 与 Ray：optimizer 和保存峰值

常规 Adam 对完整 35B 参数保留 FP32 master 和两份跨 step moment，现有约 251 GiB 主机
内存无法容纳。最终使用 file-backed FP32 master：

```text
${RUN_ROOT}/cpu_master/rank_0.bin
${RUN_ROOT}/cpu_master/rank_1.bin
总计约 121 GiB
```

文件由 `torch.from_file(..., shared=True, dtype=torch.float32)` 映射，不修改宿主机 swap，
也不依赖 root。`--ref-load` 场景把 master 初始化延迟到 distributed checkpoint 加载，
避免先从随机模型复制一遍完整 master。

优化器使用 StatelessAdam：每步更新全部参数，但不保留跨 step 的 `exp_avg` 和
`exp_avg_sq`；同时 `--no-save-optim` 不保存 optimizer state。这是让单步 smoke train
在现有资源上成立的关键取舍，不等价于标准有状态 Adam 的长期训练语义。

仅把 Ray 阈值从 95% 提到 98% 不能解决问题。`20260718_021502` 在 98% 下仍只超出约
26 MiB 就被杀。最终在 optimizer step 后解除 master 上的 `decoupled_grad` 引用并清空
`cpu_copy_map_grad`，释放两 rank 合计约 60 GiB 的 BF16 pinned staging buffer；观察到
step 后 `host_used` 从约 194 GiB 降到约 127 GiB，checkpoint 才能在保留 Ray 保护的前提
下完成。

没有采用的方案：

- 不禁用 `RAY_memory_monitor_refresh_ms`。完全关闭保护可能把机器推入系统级 OOM，诊断
  更差且可能影响其他进程。
- 不在宿主机安装或扩大 swap。file-backed master 已把大块顺序数据放到 `/data`，且符合
  “不改宿主机”的约束。
- 不保留 Adam moments。仅 moments 就会再次显著突破主机内存预算。
- 不把 FP32 master 改成 BF16。`lr=1e-6` 的逐步更新直接写回 BF16 容易被舍入掉。

### 7.3 并行布局：expert shard 必须兼容

失败布局：

```text
actor：TP=2, EP=1, ETP=2, DP=1
SGLang：TP=2, EP=2
```

它能加载 actor checkpoint，但 actor-to-SGLang 权重同步时 expert shard 维度从 256 与 512
的切分关系不一致，报错：

```text
RuntimeError: start (0) + length (512) exceeds dimension size (256)
```

最终布局：

```text
Megatron actor：TP=1, PP=1, CP=1, EP=2, ETP=1, DP=2
SGLang rollout：TP=2, EP=2, moe_dense_tp_size=1
```

该布局在两张 H200 上完成权重同步。不能只以“checkpoint 能加载”作为并行布局正确的
判据，还必须实际执行 actor-to-SGLang 更新。

### 7.4 Dynamic batching：分箱数必须与 DP 对齐

本次两条 trajectory 会形成三个约 18k-token segment。`max_tokens_per_gpu=32768` 时形成
3 个 dynamic-batch bin，但 DP=2 要求能分配为偶数个 bin，最终报：

```text
AssertionError: dynamic path: could only produce 3 mbs; need 4
```

将 `max_tokens_per_gpu` 设为 40000 后，其中两个 segment 可以合并，最终得到两个 DP bin。
该参数同时影响显存和分箱拓扑，不能只按“越小越省显存”理解。

## 8. 最终配置

```text
Actor：TP1 / PP1 / CP1 / EP2 / ETP1 / DP2
SGLang：TP2 / EP2
sglang_mem_fraction_static=0.55
sglang_max_running_requests=2
max_tokens_per_gpu=40000
log_probs_chunk_size=64
train_memory_margin_bytes=67108864
Ray object store=4 GiB
Ray memory threshold=0.98
NCCL_NVLS_ENABLE=0
gradient reduce / main grad=BF16
optimizer master=file-backed FP32
optimizer=StatelessAdam, no saved optimizer state
```

这套配置的目标是让完整机械闭环在当前双 H200 和约 251 GiB 主机内存上通过。对于多步
正式训练，StatelessAdam 的无 moments 语义、每步约 121 GiB file-backed master 扫描及
I/O 吞吐必须重新评估，不能直接把 smoke 配置视为最终训练配方。

## 9. 监控与故障判读

### 9.1 找到最新运行并看关键日志

```bash
latest="$(ls -dt /data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_* | head -1)"
tail -n 200 -f "${latest}/run.log"
```

服务器没有预装 `rg`，使用：

```bash
grep -nE \
  'CUDA out of memory|OutOfMemoryError|ActorUnavailable|could only produce|step 0:|successfully saved checkpoint|Job .* succeeded' \
  "${latest}/run.log"
```

若主日志只剩 `ActorUnavailable`，检查归档的首个 worker 异常：

```bash
grep -RniE \
  'CUDA out of memory|OutOfMemoryError|exceeds the memory usage threshold|Traceback' \
  "${latest}/ray_logs" | head -100
```

### 9.2 GPU、主机内存与 I/O

```bash
watch -n 1 nvidia-smi
watch -n 1 free -h
vmstat 1 3
```

`free -h` 中 Linux page cache 会计入 `used`，不能只看 `free`；优先看 `available`，同时
结合 Ray 日志中的 used/total 判断是否逼近 98%。`vmstat` 中持续的 `bi`/`bo` 表示
file-backed master 正在读写；StatelessAdam 扫描约 121 GiB 文件时可能约 40 分钟没有
新训练日志，只要 I/O 和进程仍活动，就不能判为卡死。

### 9.3 checkpoint 不是看见目录就算成功

```bash
cat "${latest}/checkpoints/latest_checkpointed_iteration.txt"
du -sh "${latest}/checkpoints"
grep -nE 'successfully saved checkpoint|Timer save_model end|Job .* succeeded' "${latest}/run.log"
```

保存中途也可能创建 `iter_0000000` 或部分 shard。只有 marker 为 `0`、日志明确打印
`successfully saved checkpoint`、`Timer save_model end`，并且 Ray Job succeeded，才能
认定 checkpoint 完整。失败运行中的 partial checkpoint 不应作为续训输入。

### 9.4 临时 cpu_master

```bash
du -sh "${latest}/cpu_master" 2>/dev/null || true
ls -lh "${latest}/cpu_master" 2>/dev/null || true
```

训练过程中出现约 121 GiB 的 `rank_0.bin` 和 `rank_1.bin` 是预期行为。启动脚本的 EXIT
trap 会在成功或失败退出时删除它们。运行结束后目录仍存在，说明进程被强杀、trap 未执行
或仍有任务存活；先确认没有训练进程和 Ray job，再做人工清理，不能在训练中删除映射文件。

## 10. 最终成功证据

```text
运行目录：/data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_20260718_031727
Ray job：raysubmit_gyexLHDqA4QXQHLf
remote rollouts：2 条，reward=1.00，applied=True，agent_exit_code=0
actor：forward/backward 完成
训练：step 0 打印完成
checkpoint：iteration 0 successfully saved
checkpoint marker：0
checkpoint 大小：65 GiB
Timer save_model end：374.2s
Ray Job succeeded
cpu_master：退出后已清理
```

训练指标：

```text
train/loss=0.0
train/pg_loss=0.0
train/entropy_loss=0.10291240073704605
train/grad_norm=0.0
train/lr=1e-6
actor_train_time=2048.1s
actor_train_tok_per_s=27.085979450224347
```

本 smoke 数据只有一个 calculator 修复任务，两条 trajectory 都得到 reward 1，因此 GRPO
组内 advantage 为 0，继而 policy loss 和 grad norm 为 0。这不是训练链路未执行；actor
forward/backward、optimizer step 和 checkpoint 均有独立日志证据。要验证非零更新，必须
换成能产生组内 reward 差异的数据，不应通过篡改成功验收指标制造非零梯度。

## 11. 可迁移经验

1. 先用 rollout-only 切开推理/sandbox 问题与训练问题，避免同时调两条链路。
2. 对 MoE 训练按完整参数量估算权重和 optimizer 状态，不按 active parameter 估算。
3. CUDA OOM 要记录失败申请大小和发生阶段；34 GiB、486 MiB 和 122 MiB 指向完全不同的
   张量生命周期问题。
4. Ray worker 消失先查第一个异常，RPC closed 和 broken pipe 大多只是级联结果。
5. 并行布局必须同时通过 checkpoint load、训练和 actor-to-rollout 权重同步。
6. 提高内存阈值只能留出峰值空间，不能代替释放已无用的 tensor 引用。
7. checkpoint 的验收必须包含 marker、完整日志、目录大小和 Ray 最终状态。
8. 所有 35B 特化放在专用入口中，保留已验证的 4B 回退路径。
