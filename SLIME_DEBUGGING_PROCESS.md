# slime Coding Agent 本机 Docker 调试全过程

## 1. 目标与最终结果

目标是在单机上跑通 `examples/coding_agent_rl`，采用以下路线：

- 本地 Docker 代替 E2B 作为 agent 和 clean-eval 沙箱。
- 使用本机已有的 Qwen3.5-4B。
- 使用 Claude Code 作为 coding harness，但通过本地 Anthropic adapter 调用 Qwen，
  不使用 Anthropic API Key。
- 先验证 sandbox 和 rollout，再执行一次单步 GRPO 训练。
- 只使用物理 GPU 1，不运行仓库原有的多机脚本。

2026-07-17 最终结果：

- Docker 内 agent 测试：`65 passed, 1 skipped`。
- 最终 rollout：`reward=1.00`、`applied=True`、`agent_exit_code=0`。
- 单步 GRPO：两条轨迹均通过 clean eval，完成 forward、backward、optimizer 和
  checkpoint 保存。
- checkpoint 大小约 `55GB`，训练 Ray job 正常退出。
- 两条轨迹奖励都为 1，组内标准差为 0，因此 advantage、loss 和 grad norm 都是
  0。本次证明的是训练链路完整可运行，不代表发生了非零参数更新。

## 2. 初始环境盘点与方案选择

仓库最初检出状态：

- 路径：`/home/xys/slime`
- 分支：`main`
- 基准提交：`fb42ae45`

原始 coding-agent 示例不能直接在本机运行：

- 默认脚本面向 `8 节点 x 8 GPU`、Qwen3.6-35B-A3B 和 96K 上下文。
- 原实现依赖 E2B 兼容 sandbox 服务。
- 本机有 `2 x H200 NVL 141GB`，调试时 GPU 0 已被其他任务占用，GPU 1 空闲。
- 本机已有 `/home/xys/models/Qwen3.5-4B`，但没有对应的 Megatron
  `torch_dist` checkpoint。
- 当时 `/home` 剩余空间约 232GB，大模型资产和运行结果因此放到
  `/data/xys/slime-coding-agent`。

据此将目标缩小为单 GPU 的最小闭环：

```text
宿主机
├── slime-coding-agent-main       GPU 1；Ray、SGLang、Megatron、adapter
├── slime-agent-<id>              临时 agent sibling container
└── slime-agent-<id>              全新 clean-eval sibling container
```

主容器挂载宿主机 `/var/run/docker.sock`，通过 Docker CLI 创建 sibling
container。所有容器加入 `slime-coding-agent-net`，任务容器通过主容器名和
`18001` 端口访问 adapter。

## 3. 实现内容

### 3.1 DockerSandbox

在 `slime/agent/sandbox.py` 中增加：

- `DockerSandbox`：支持容器创建、命令执行、文件读写和退出清理。
- `create_sandbox()`：根据 `SLIME_AGENT_SANDBOX_BACKEND` 选择 E2B 或 Docker。
- `write_file(..., user=...)`：先以 root 写入，再按需 `chown` 给 agent 用户。
- 对本地 `Path` 使用 `docker cp --follow-link`，复制符号链接指向的实际文件。

`examples/coding_agent_rl/generate.py` 和
`examples/coding_agent_rl/swe.py` 改为通过 factory 创建工作沙箱及评测沙箱，
保留 E2B 作为默认后端。

### 3.2 本地运行资产

新增 `examples/coding_agent_rl/local_docker/`，其中包括：

- `Dockerfile.runtime`：以 slime runtime 为基础，补入 Docker CLI。
- `Dockerfile.swe_smoke`：构建一个带有预置加法 bug 和 pytest 的最小任务镜像。
- `smoke.jsonl`：单条标准 message-list prompt 及 clean-eval metadata。
- `run_host.sh`：统一启动 test、sandbox-smoke、claude-smoke、convert、rollout、train。
- `run_qwen35_4b.sh`：单 GPU Ray/SGLang/Megatron 参数。
- `convert_qwen35_4b.sh`：HF checkpoint 到 Megatron `torch_dist` 的转换。
- `sandbox_smoke.py`：真实 sibling container 生命周期、diff 和 clean eval 集成检查。
- `claude_smoke.py`：Node/Claude Code 安装及 `claude --version` 检查。

新增 `tests/test_agent/test_docker_sandbox.py`，并调整 fake sandbox 测试以兼容
factory 和 `user` 参数。

### 3.3 运行时约束

本地脚本最终采用以下关键设置：

- 宿主机仅映射 `--gpus device=1`；容器内该设备显示为 GPU 0。
- `--shm-size=32g`。
- `--ulimit nofile=1048576:1048576`。
- Ray 限制为 `--num-cpus 8 --num-gpus 1`。
- Qwen 上下文 32768，最大响应 4096。
- SGLang 最大并发请求 4，静态显存比例 0.65。
- 单卡 colocate，optimizer CPU offload。
- rollout 模式采样 1 条；train 模式同一 prompt 采样 2 条。

## 4. 资产和镜像准备

准备的 harness 资产：

- Node.js `22.20.0` Linux x64，约 30MB。
- Claude Code `2.1.212` npm 包，约 23KB。该 npm 包是安装器，平台二进制在
  sandbox 中安装时准备。

校验和：

```text
00bbd05e306ea68b6e13e17360d0e2f680b493ef95f2fea1c4296ff7437530bc  node-v22-linux-x64.tar.xz
2162841dd793d21671eccb7fe76fe9c3da6816adf447ba3890a4871b7e5f4e69  anthropic-ai-claude-code-2.1.212.tgz
```

构建后的本地镜像：

```text
slime-coding-agent-runtime:local
  image id/digest: sha256:b327465cfd4b73a48c7f1dc0e1e48be9c8cef8dd7d4c2820dfead5b915c95593
  size: 21.5GB

slime-coding-agent-swe-smoke:local
  image id/digest: sha256:172e4c1e1e87b2740c4c72fa01a01bd859204f758aec930cd45d88eee2f7a402
  size: 88MB
```

Qwen3.5-4B 转换后的 checkpoint：

```text
/data/xys/slime-coding-agent/models/Qwen3.5-4B_torch_dist
size: 7.9GB
latest_checkpointed_iteration.txt: release
```

## 5. 调试时间线

### 5.1 基础 sandbox 和 harness

1. 新增 Docker backend 后，Python 编译和不需要监听端口的单测通过。
2. 首次真实 sandbox smoke 已能创建 sibling container、执行命令和自动清理，
   但任务镜像内 pytest 无法导入 `calculator`。
3. 在 smoke repo 的 pytest 配置中加入正确的 Python path 后，预置测试能按预期
   失败，证明任务 fixture 有效。
4. Claude Code 首次安装报 npm `EACCES`。完整 npm 日志显示，稳定文件名
   `anthropic-ai-claude-code.tgz` 是宿主机符号链接，`docker cp` 原样保留了链接，
   但 sibling container 中不存在链接目标。
5. 将 `Path` 复制改为 `docker cp --follow-link`，并在 npm 安装前显式
   `chmod 0644 /tmp/harness-cli.tgz`。随后 `claude --version` 返回 `2.1.212`。

### 5.2 权重转换

使用 runtime 容器和 `scripts/models/qwen3.5-4B.sh` 中的模型参数，将本机 HF
Qwen3.5-4B 转换为 Megatron `torch_dist`。转换成功，产物约 7.9GB，两个权重
分片完整，iteration 标记为 `release`。

### 5.3 rollout 失败与修复

以下运行目录保留了每次尝试。表中的“修复”是该次失败后采取的动作。

| 运行目录 | 现象/根因 | 修复 |
| --- | --- | --- |
| `rollout_20260717_083702` | 在提交 Ray job 前失败。容器强制使用 loopback，但 Ray 节点注册为 Docker 网卡地址。 | `MASTER_ADDR` 自动取容器 `eth0` 地址；等待 dashboard/job agent 就绪。 |
| `rollout_20260717_083907` | dashboard 回连 Docker 网卡上的 job agent 不稳定，代理绕过配置不完整。 | 把节点地址加入 `no_proxy`，启动 Ray 前清空 HTTP(S) proxy。 |
| `rollout_20260717_084018` | 排除代理后仍出现相同的 job-agent 错误，仅凭 CLI 表象无法继续判断。 | 失败时把 `/tmp/ray/session_latest/logs` 归档到运行目录。 |
| `rollout_20260717_084118` | Ray 日志显示 `Too many open files`；容器继承宿主机 384 CPU，Ray 预建大量 worker，raylet 最终 `SIGABRT`。 | Ray 限制为 8 CPU；主容器 `nofile` 提升到 1,048,576。 |
| `rollout_20260717_084504` | Job 已提交并进入 `train.py`，但 Qwen3.5-4B 是 `ForConditionalGeneration`，processor 要求 prompt 为消息列表而非字符串。 | 将 `smoke.jsonl` 的 prompt 改成标准 user message list。 |
| `rollout_20260717_084730` | SGLang 和 adapter 已连通，但 Claude Code 的系统提示和工具 schema 约 17.6K tokens，超过 16K 上限。 | 上下文提高到 32K；响应仍限制为 4K；SGLang 并发降到 4。 |
| `rollout_20260717_085051` | Claude Code 已生成非空源码 diff，但 clean-eval 无法把 patch 文件写入 root 拥有的 `/workspace`。 | Docker backend 以 root 写文件，然后 `chown agent:agent`。 |
| `rollout_20260717_085501` | Agent 内测试通过，但 clean eval `applied=False`、reward 0。 | 给 `_apply_diff` 增加逐种 apply 方法的退出码和错误尾部。 |
| `rollout_20260717_090308` | 实际 diff 含 pytest 生成的未跟踪 `__pycache__/*.pyc`；不完整二进制 patch 无法应用。第一次 `--3way` 还污染了工作树，使 fallback 报 index 不匹配。 | diff pathspec 排除 pyc、`__pycache__`、`.pytest_cache`；所有 fallback 改成先 check/dry-run 再 apply。 |
| `rollout_20260717_090950` | 最终验证通过。 | `reward=1.00`、`applied=True`、`agent_exit_code=0`。 |

在最后一次修复期间，增强后的 `sandbox-smoke` 首次又暴露直接执行脚本时
`sys.path` 只有 `local_docker/`。脚本显式加入仓库根目录后，以下集成路径通过：

```text
运行 pytest 产生缓存
  -> 只捕获源码 diff
  -> 在全新 sibling container 中应用 diff
  -> clean eval 返回 1
```

### 5.4 最终 rollout

运行目录：

```text
/data/xys/slime-coding-agent/runs/rollout_20260717_090950
```

关键日志：

```text
[coding_agent_rl] local-calculator-001: reward=1.00 applied=True agent_exit_code=0 elapsed=22.2s segments=1
rollout/response_len/mean: 418.0
Job 'raysubmit_kapAUg1YRAE3XEPn' succeeded
```

Claude Code 使用 Read/Bash/Edit 等工具把 `return left - right` 修成
`return left + right`。修改在独立 clean-eval container 中重新应用并运行
`pytest -q`，不是沿用 agent 的工作容器。

### 5.5 单步 GRPO

运行目录：

```text
/data/xys/slime-coding-agent/runs/train_20260717_091310
```

配置和结果：

- `n_samples_per_prompt=2`
- `global_batch_size=2`
- `num_rollout=1`
- `num_steps_per_rollout=1`
- 两条轨迹均 `reward=1.00 applied=True agent_exit_code=0`
- actor 的约 4.2B 参数及转换后的 checkpoint 加载成功
- step 0 完成，学习率 `1e-6`
- checkpoint 保存耗时约 310.4 秒
- Ray job `raysubmit_92PxGmEHeBBiw89i` succeeded

训练指标中的关键限制：

```text
rollout/raw_reward: 1.0
rollout/advantages: 0.0
train/loss: 0.0
train/grad_norm: 0.0
```

这是因为两条 smoke 轨迹奖励相同。链路包含反向传播和 optimizer step，但没有
非零梯度更新。若要验证学习行为，需要至少构造一个能产生组内奖励差异的任务或
采样设置。

## 6. 最终产物

```text
/data/xys/slime-coding-agent/
├── assets/
├── models/Qwen3.5-4B_torch_dist/                 约 7.9GB
└── runs/
    ├── rollout_20260717_090950/
    │   ├── run.log
    │   └── rollout_dumps/rollout_0.pt
    └── train_20260717_091310/
        ├── run.log
        ├── rollout_dumps/
        └── checkpoints/                          约 55GB
            ├── latest_checkpointed_iteration.txt  内容为 0
            ├── iter_0000000/
            └── rollout/
```

## 7. 当前代码状态

这些修改尚未提交。仅检出上游基准提交 `fb42ae45` 不能复现本次结果，必须保留
当前工作树的修改和未跟踪文件。

主要修改文件：

```text
examples/coding_agent_rl/generate.py
examples/coding_agent_rl/swe.py
slime/agent/harness/common.py
slime/agent/sandbox.py
tests/test_agent/_fakes.py
tests/test_agent/test_agent_rollout_cpu.py
```

主要新增文件：

```text
examples/coding_agent_rl/local_docker/
tests/test_agent/test_docker_sandbox.py
SLIME_DEBUGGING_PROCESS.md
SLIME_LOCAL_REPRODUCTION.md
```

## 8. 经验与风险

- 不应直接运行原始 64 GPU 脚本，其中包含面向集群环境的全局进程操作。
- Ray 在大 CPU 宿主机内必须显式限制逻辑 CPU 数，并提高文件描述符上限。
- 对 processor 模型，数据 prompt 的结构和 tokenizer-only 模型不同。
- Coding harness 的固定系统提示可能显著占用上下文，16K 对 Claude Code 不够。
- clean eval 必须使用全新容器，并确保失败的 patch fallback 不会污染工作树。
- diff 捕获必须排除运行测试产生的二进制缓存和临时目录。
- 挂载 `/var/run/docker.sock` 等价于向主容器授予宿主机 Docker/root 级能力，
  只能运行可信代码与可信任务镜像。

