# slime 双服务器架构说明

## 1. 文档范围

本文说明当前 slime Qwen3.5-35B-A3B remote Docker 方案中两台服务器的职责、组件边界、
网络链路、数据流、生命周期和故障域。

本文只描述架构，不替代操作手册和调试记录：

```text
可复现操作：SLIME_REMOTE_DOCKER_35B_A3B_REPRODUCTION.md
完整调试记录：SLIME_QWEN35_35B_A3B_DEBUGGING_PROCESS.md
原单机流程：SLIME_LOCAL_REPRODUCTION.md
```

文档不包含密码或私钥内容。服务器间只使用专用 SSH 公钥认证。

## 2. 一句话概括

`.102` 是 GPU 计算和持久化中心，负责模型加载、LLM rollout、训练、日志与 checkpoint；
`.101` 是隔离执行节点，只负责为 coding agent 创建临时 task/eval Docker 容器。两台机器
不共享文件系统，通过 SSH 转发的 Docker Unix socket 下发容器操作，并通过
`.101 -> .102:18001` 的 HTTP 回调让远端 agent 访问模型 adapter。

## 3. 总体拓扑

```text
开发终端
  E:\Data\llm\.codex-ssh-xys-110-102.conf
        |
        | SSH 22，public-key only
        v
+--------------------------------------------------------------------------------+
| 192.168.110.102 / master                                                       |
| 用户：xys                                                                      |
| 角色：GPU runtime、模型、训练、持久化                                            |
|                                                                                |
| /home/xys/slime                                                               |
| /data/xys/slime-coding-agent                                                  |
| 2 x NVIDIA H200 NVL                                                           |
|                                                                                |
| Docker: slime-coding-agent-runtime:local                                      |
|   container: slime-coding-agent-main-remote-35b-a3b                           |
|   network: slime-coding-agent-main-net                                        |
|                                                                                |
|   Ray head + workers                                                          |
|      +-- Megatron actor: TP1 / EP2 / DP2                                      |
|      +-- SGLang engine: TP2 / EP2                                             |
|      +-- coding-agent adapter: 0.0.0.0:18001                                  |
|      +-- Docker CLI                                                           |
|             |                                                                  |
|             | DOCKER_HOST=unix:///var/run/slime-remote-docker.sock            |
|             v                                                                  |
|   mounted temporary Unix socket                                                |
+-------------------------------+------------------------------------------------+
                                |
                                | SSH 22，Unix-socket forwarding
                                | .102 temporary socket -> .101 /var/run/docker.sock
                                v
+--------------------------------------------------------------------------------+
| 192.168.110.101 / cu01                                                         |
| 用户：whz，属于 docker 组                                                      |
| 角色：DockerSandbox、task container、clean-eval container                      |
|                                                                                |
| Docker Engine 24.0.9                                                          |
| /var/run/docker.sock                                                          |
| network: slime-coding-agent-net                                               |
| image: slime-coding-agent-swe-smoke:local                                     |
|                                                                                |
| temporary task/eval container                                                 |
|      +-- 安装/运行 coding agent                                                |
|      +-- 修改任务工作树                                                        |
|      +-- clean container 中执行 pytest                                         |
|      +-- HTTP -> 192.168.110.102:18001 -> adapter -> SGLang                   |
+--------------------------------------------------------------------------------+
```

两台机器上的 Docker network 是两个独立的本地 bridge network，并没有组成 overlay
network。跨机回调使用物理网络地址 `192.168.110.102:18001`，不能使用 `.102` runtime
容器名访问。

## 4. 服务器职责

| 资源或组件 | `.102` / master | `.101` / cu01 |
|---|---|---|
| Qwen3.5-35B-A3B HF 权重 | 保存并只读挂载进 runtime | 不保存 |
| Megatron distributed checkpoint | 保存于 `/data` | 不保存 |
| H200 GPU | 两张，负责 rollout 和训练 | 不使用 GPU 参与本流程 |
| Ray | head、job agent、actor 均在 runtime 内 | 不运行 |
| SGLang | 双卡 TP2/EP2，生成模型 rollout | 不运行 |
| Megatron actor | 双卡 TP1/EP2/DP2，执行训练 | 不运行 |
| coding-agent adapter | 运行并发布 `:18001` | 通过 HTTP 访问 |
| runtime image | `slime-coding-agent-runtime:local` | 不需要 |
| task image | 保留一份用于校验和同步 | `slime-coding-agent-swe-smoke:local` |
| Docker daemon | 启动主 runtime 容器 | 启动 task/eval 容器 |
| 训练数据与运行日志 | 保存 | 不保存 |
| rollout dump 与 checkpoint | 保存 | 不保存 |
| 临时 sandbox 工作树 | 不直接保存 | 位于临时容器，退出时删除 |

架构上不存在“把模型 rollout 推理迁移到 `.101`”。迁移到 `.101` 的是 rollout 过程中
coding agent 的工具执行和验证容器；LLM token generation 仍由 `.102` 的 SGLang 完成。

## 5. `.102` GPU/runtime 节点

### 5.1 宿主机资源

```text
主机名：master
地址：192.168.110.102
用户：xys
GPU 0：NVIDIA H200 NVL
GPU 1：NVIDIA H200 NVL
Docker Engine：29.6.1
仓库：/home/xys/slime
工作区：/data/xys/slime-coding-agent
```

35B 入口把物理 GPU `0,1` 映射进 runtime，并在容器内设置
`CUDA_VISIBLE_DEVICES=0,1`。两卡之间当前不使用 NCCL NVLS：

```text
NCCL_NVLS_ENABLE=0
```

### 5.2 本地 runtime 容器

```text
image：slime-coding-agent-runtime:local
container：slime-coding-agent-main-remote-35b-a3b
network：slime-coding-agent-main-net
IPC：host
shared memory：32 GiB
adapter bind：0.0.0.0:18001
host publish：192.168.110.102:18001:18001
```

主要挂载：

| `.102` 宿主机路径 | runtime 容器路径 | 用途 |
|---|---|---|
| `/home/xys/slime` | `/root/slime` | 代码与启动脚本 |
| `/data/xys/slime-coding-agent` | `/workspace` | 运行目录、资产、转换模型、日志、checkpoint |
| HF 模型目录 | `/models/Qwen3.5-35B-A3B` | SGLang/HF 配置和权重，只读 |
| 临时 SSH tunnel socket | `/var/run/slime-remote-docker.sock` | 访问 `.101` Docker API |

SSH 私钥不会挂载进 runtime。外层宿主机脚本先建立 tunnel，runtime 只看到 tunnel 的
Unix socket，因此容器内的 Docker CLI 可以操作 `.101` daemon，但不能直接读取专用私钥。

### 5.3 Ray、SGLang 与 actor

训练脚本在 runtime 内启动单节点 Ray：

```text
Ray head port：6379（容器内）
Ray dashboard/job agent：8265（容器内）
Ray CPU：16
Ray GPU：2
Ray object store：4 GiB
Ray memory usage threshold：0.98
```

Ray 内部创建两类 GPU worker：

```text
SGLang rollout
  TP=2, EP=2
  mem_fraction_static=0.55
  max_running_requests=2

Megatron actor
  TP=1, PP=1, CP=1
  EP=2, ETP=1, DP=2
```

两者 colocate 在相同 H200 上，通过 sleep/wake 复用显存。SGLang 负责采样，Megatron
负责 forward、backward 和 optimizer step，actor 更新后的权重再同步回 SGLang。

## 6. `.101` sandbox 节点

### 6.1 宿主机与权限

```text
主机名：cu01
地址：192.168.110.101
运行用户：whz
Docker Engine：24.0.9，API 1.43
containerd：1.6.33
Docker socket：/var/run/docker.sock
Docker network：slime-coding-agent-net
```

`whz` 属于 `docker` 组，可以通过 Unix socket 使用 Docker daemon。Docker 组权限等价于
该节点上的 root 级容器控制权限，因此 `.102` 到 `.101` 的专用 SSH 私钥必须只用于此
受控链路，不能复制到镜像、sandbox 或日志。

### 6.2 远端容器

`.101` 只需要任务镜像，不需要 runtime image、模型权重、Ray 或 CUDA 训练环境：

```text
slime-coding-agent-swe-smoke:local
```

每条 trajectory 典型涉及：

1. task container：coding agent 在任务工作树中检查代码并生成修改。
2. clean-eval container：在干净镜像状态中应用 patch 并执行 pytest。
3. harness 收集 source diff、测试结果、`applied`、`agent_exit_code` 和 reward。
4. 正常或异常退出后删除本次临时容器。

Node 和 Claude Code 安装包存放在 `.102:/data/xys/slime-coding-agent/assets`。它们由
runtime 内的 Docker CLI 通过远端 Docker API 复制进 `.101` 临时容器，不要求 NFS、
Samba 或 `/data` 共享挂载。

## 7. 三条跨机通信链路

### 7.1 控制面：SSH 公钥

```text
.102 xys
  /home/xys/.ssh/slime_docker_whz_192_168_110_101_ed25519
       |
       | SSH 22
       v
.101 whz
```

固定安全选项包括：

```text
IdentitiesOnly=yes
BatchMode=yes
PasswordAuthentication=no
StrictHostKeyChecking=yes
UserKnownHostsFile=/home/xys/.ssh/known_hosts
ExitOnForwardFailure=yes
```

### 7.2 Docker 控制：Unix-socket tunnel

外层脚本在 `.102` 创建每次运行独立的临时 socket：

```text
/data/xys/slime-coding-agent/remote-docker/docker-${PID}.sock
```

SSH forwarding 关系：

```text
.102 docker-${PID}.sock
       -> encrypted SSH connection
       -> .101 /var/run/docker.sock
```

runtime 中的映射：

```text
/var/run/slime-remote-docker.sock
DOCKER_HOST=unix:///var/run/slime-remote-docker.sock
```

没有开放未加密的 Docker TCP `2375`，也没有把 `.101:/var/run/docker.sock` 通过网络文件
系统直接暴露。`.102` Docker CLI 29.6.1 与 `.101` daemon 24.0.9 通信时把 API 降级到
双方兼容的 1.43。

### 7.3 数据面：agent 到模型 adapter

```text
.101 task container
       |
       | HTTP 192.168.110.102:18001
       v
.102 host published port :18001
       |
       v
runtime adapter
       |
       v
SGLang engine -> H200 GPU 0,1
```

`.101` 必须能路由到 `.102:18001`。任何 HTTP 响应都说明网络路径已建立；连接超时通常是
路由或防火墙，connection refused 通常是 runtime/adapter 尚未监听。

## 8. 数据与制品位置

### 8.1 `.102` 持久化内容

```text
/home/xys/slime
  代码、35B wrapper、CPU offload hook、Markdown 文档

/home/xys/ms-swift/model/Qwen/Qwen3.5-35B-A3B
  Hugging Face checkpoint

/data/xys/slime-coding-agent/models/Qwen3.5-35B-A3B_torch_dist
  Megatron torch distributed checkpoint，约 65 GiB

/data/xys/slime-coding-agent/assets
  Node 与 Claude Code 离线资产

/data/xys/slime-coding-agent/runs/<RUN_ROOT>
  run.log
  rollout_dumps/
  checkpoints/
  ray_logs/（失败时归档）
  cpu_master/（训练期间临时存在）
```

训练时 `${RUN_ROOT}/cpu_master/rank_0.bin` 和 `rank_1.bin` 合计约 121 GiB，是 file-backed
FP32 optimizer master。它们只在 `.102:/data` 上创建，正常退出时由 trap 删除，不复制到
`.101`。

### 8.2 `.101` 持久化内容

```text
Docker Engine 与 containerd 数据
slime-coding-agent-swe-smoke:local image
slime-coding-agent-net network
```

task/eval 容器和任务工作树是临时内容。训练数据、模型、rollout dump、checkpoint 和
CPU master 不落盘到 `.101`。

### 8.3 镜像同步

`.101` 不要求访问 Docker Hub。任务镜像从 `.102` 通过现有 tunnel 同步：

```text
.102 docker save slime-coding-agent-swe-smoke:local
       |
       | stream through remote Docker client/tunnel
       v
.101 docker load
```

同步后比较两端完整 RootFS layer digest。runtime image 不同步，因为它始终只在 `.102`
运行。

## 9. 一次训练的端到端时序

```text
1. 用户在 .102 执行 run_remote_docker_35b_a3b.sh train。
2. wrapper 用专用公钥登录 .101，检查 Docker 和磁盘。
3. .102 建立临时 Unix-socket SSH tunnel，并通过 /_ping 验证远端 daemon。
4. .102 Docker daemon 启动本地 35B runtime 容器，挂载代码、模型、workspace 和 tunnel。
5. runtime 启动 Ray head、Megatron actor 和 SGLang engine。
6. coding-agent harness 通过 DOCKER_HOST 请求 .101 创建 task container。
7. .101 task container 通过 .102:18001 调用 adapter；SGLang 在双 H200 上生成 token。
8. agent 在多轮模型交互中检查并修改 task container 内的代码。
9. .101 创建 clean-eval container，应用 patch、运行 pytest、返回 reward。
10. 两条 trajectory 回到 .102，Megatron 执行 forward/backward/optimizer。
11. actor 权重同步给 SGLang；iteration 0 checkpoint 写入 .102:/data。
12. Ray job 结束，runtime --rm 删除；脚本关闭 SSH tunnel 并删除临时 socket。
13. sandbox harness 删除 .101 的本次 task/eval 容器。
14. .102 EXIT trap 删除本次约 121 GiB 的 cpu_master 文件。
```

## 10. 端口、socket 与网络清单

| 端点 | 方向 | 用途 | 暴露范围 |
|---|---|---|---|
| `.102:22` | 开发终端 -> `.102` | 运维 SSH | 局域网 SSH |
| `.101:22` | `.102` -> `.101` | SSH 控制与 socket forwarding | 局域网 SSH |
| `.102:18001` | `.101` task container -> `.102` | agent adapter 回调 | 绑定 `.102` 指定地址 |
| `.101:/var/run/docker.sock` | SSH tunnel -> `.101` | Docker daemon API | 仅本机 Unix socket |
| `.102:/data/.../docker-${PID}.sock` | wrapper/runtime | tunnel 本地端 | `.102` 用户目录，临时 |
| runtime `:6379` | Ray 进程内部 | Ray head | runtime 内部 |
| runtime `:8265` | wrapper/job submit 内部 | Ray dashboard/job agent | runtime 内部 |
| runtime SGLang `:15000` 等 | Ray/SGLang 内部 | engine RPC/HTTP | runtime 内部 |

除 adapter `18001` 外，Ray、SGLang 和 Docker API 都不需要作为跨机 TCP 服务暴露。

## 11. 生命周期与清理边界

### 11.1 每次运行创建并清理

```text
.102 runtime container（--rm）
.102 SSH tunnel 进程
.102 docker-${PID}.sock
.102 cpu_master/rank_*.bin
.101 task/eval containers
```

### 11.2 跨运行保留

```text
.102 runtime/task images
.102 HF 与转换 checkpoint
.102 run.log、rollout dump、训练 checkpoint
.101 task image
.101 slime-coding-agent-net
```

不要使用宽泛的 `docker system prune` 或批量 `docker rm`。`.101` 可能承载其他任务；仅清理
确认带 `slime.agent.sandbox=true` 标签、名称属于本次运行的容器。

## 12. 故障域与影响

| 故障位置 | 典型现象 | 影响范围 | 首要检查 |
|---|---|---|---|
| `.102` GPU/CUDA | CUDA OOM、SGLang/actor 退出 | rollout 生成或训练失败 | `nvidia-smi`、首个 Ray worker traceback |
| `.102` 主机内存 | Ray memory monitor 杀 worker | optimizer 或 checkpoint 失败 | Ray used/total、`free -h`、staging gradient |
| `.102:/data` | 空间不足、I/O 很慢 | 模型转换、cpu_master、checkpoint | `df -h /data`、`vmstat` |
| `.102` runtime daemon | runtime 无法启动 | 整条流程不能进入 Ray | 本地 `docker version`、image inspect |
| `.102 -> .101` SSH | socket 不生成、keepalive 失败 | 无法创建 sandbox | key、known_hosts、22 端口、SSH 日志 |
| `.101` Docker daemon | `_ping`/Docker API 失败 | sandbox/eval 不可用 | `systemctl status docker`、socket 权限 |
| `.101` 磁盘 | pull/load/run 失败 | 新容器无法创建 | `df -h /var/lib/docker` |
| `.101 -> .102:18001` | timeout/refused | agent 无法调用模型 | adapter 监听、端口发布、路由/防火墙 |
| task image 缺失 | metadata image not found | 对应任务启动失败 | `.101 docker image inspect`、执行 build 同步 |

`.101` 故障不会破坏 `.102` 已有模型或 checkpoint，但当前流程没有 sandbox 节点冗余，
因此会阻断新的 trajectory。`.102` 故障则同时影响模型 rollout、训练和所有持久化结果，
它是当前架构的主故障域。

## 13. 安全边界

1. 开发终端到 `.102`、`.102` 到 `.101` 都固定使用公钥，禁用密码回退。
2. `.101` Docker daemon 不监听 `0.0.0.0:2375`，只保留本地 Unix socket。
3. SSH 私钥只存在于 `.102` 宿主机，不进入 runtime/task image 或训练日志。
4. runtime 只得到单次运行的 tunnel socket；脚本退出后 socket 失效并被删除。
5. `.101` 的 docker 组权限等价于 root，只允许可信 image、harness 和受控数据集使用。
6. task metadata 指定的所有 image 必须预先审核并同步到 `.101`。
7. adapter `18001` 是当前唯一必要的跨机应用端口，应限制在可信局域网范围。
8. 文档和命令中不保存账号密码、私钥正文或访问令牌。

## 14. 当前在线快照

以下信息于 2026-07-18 从 35B wrapper 的 `preflight` 和只读系统命令实测：

```text
.102
  hostname=master
  GPU=2 x NVIDIA H200 NVL
  Docker client/server=29.6.1 / API 1.55
  /data=7.3 TiB total, 5.4 TiB available
  / 所在文件系统=1.8 TiB total, 144 GiB available

.101
  hostname=cu01
  user=whz, groups include docker
  Docker client/server=24.0.9 / API 1.43
  containerd=1.6.33
  Docker 所在文件系统=847 GiB total, 396 GiB available

跨机
  .102 Docker CLI -> .101 Docker daemon：API 自动降级为 1.43
  SSH Unix-socket tunnel /_ping：OK
```

容量是时间点快照，不是固定配额。35B 训练在 `.102:/data` 还需要模型约 65 GiB、临时
FP32 master 约 121 GiB 和每次训练 checkpoint 约 65 GiB；执行前仍必须重新检查磁盘。

## 15. 已验证的最终运行

```text
RUN_ROOT=/data/xys/slime-coding-agent/runs/train_qwen35_35b_a3b_20260718_031727
Ray job=raysubmit_gyexLHDqA4QXQHLf
2 remote rollouts：reward=1.00, applied=True, agent_exit_code=0
actor forward/backward/optimizer：完成
checkpoint iteration 0：保存成功，约 65 GiB
Ray Job succeeded
cpu_master：退出后已清理
```

该运行同时验证了 `.102` 双 H200 模型生成与训练、`.102 -> .101` Docker 控制链路、
`.101 -> .102:18001` agent 回调、clean eval、reward 回传和 checkpoint 持久化。

## 16. 回退与演进边界

跨机脚本是新增入口，原 `.102` 单机 Docker 方案仍保留。跨机 runtime 和 tunnel 完全退出
后，可使用原 `run_host.sh` 回退；无需卸载 `.101` Docker，也无需修改模型 checkpoint。

当前架构的主要扩展边界：

- 增加正式数据集时，必须先把每个 `metadata.image` 同步到 `.101`。
- 增加 sandbox 并发前，需评估 `.101` CPU、内存、Docker 磁盘和 `.102:18001` 并发。
- 增加第二个 sandbox 节点需要新增调度和按任务选择 `DOCKER_HOST`，当前脚本只指向 `.101`。
- 多节点 GPU 训练需要重新设计 Ray/NCCL 网络；当前架构是 `.102` 单节点双 GPU。
- 多步正式训练需要重新评估 StatelessAdam 无 moments 语义和 `/data` I/O，当前成功配置
  首先面向完整单步机械闭环。
