# slime 训练与 Docker rollout 跨机复现流程

## 1. 目标与回退边界

本流程把计算分成两部分：

- `192.168.110.102`：保留 slime runtime、Ray、SGLang、Megatron、模型、训练
  数据、rollout dump 和 checkpoint，继续使用物理 GPU 1。
- `192.168.110.101`：只运行 coding-agent rollout 创建的 task container 和
  clean-eval container。

原单机流程未被覆盖：

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh rollout
bash examples/coding_agent_rl/local_docker/run_host.sh train
```

跨机流程使用独立入口：

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh MODE
```

`SLIME_LOCAL_REPRODUCTION.md` 和 `run_host.sh` 保持原样。跨机脚本退出后，直接执行
原命令即可回到 `.102` 单机 Docker 方案。

## 2. 架构与数据流

```text
192.168.110.102 (master, GPU/runtime)
  local slime runtime container
    Ray + SGLang + Megatron + adapter :18001
    Docker CLI
       |
       | local Unix socket mounted into runtime
       v
  xys-owned SSH Unix-socket tunnel
       |
       | encrypted SSH, public-key only
       v
192.168.110.101 (cu01, sandbox)
  /var/run/docker.sock
    task container + fresh clean-eval container
       |
       +---- http://192.168.110.102:18001 ----> adapter ----> SGLang
```

没有开放 Docker TCP `2375`。Docker daemon 只监听 `.101:/run/docker.sock`，临时
转发 socket 位于 `.102:/data/xys/slime-coding-agent/remote-docker`，文件名包含启动
进程 PID，并由退出 trap 清理。

## 3. `.101` Docker 安装记录

`.101` 原先没有 Docker。2026-07-17 经明确授权安装以下 Docker 官方 CentOS 7
签名 RPM：

```text
docker-ce                 24.0.9-1.el7
docker-ce-cli             24.0.9-1.el7
docker-ce-rootless-extras 24.0.9-1.el7
containerd.io             1.6.33-3.1.el7
docker-buildx-plugin      0.14.1-1.el7
docker-compose-plugin     2.24.7-1.el7
```

同时安装了 RPM 明确要求的 CentOS 7 依赖：

```text
container-selinux 2.119.2
fuse-overlayfs   0.7.2
fuse3-libs       3.6.1
slirp4netns      0.4.3
```

Docker Release GPG 指纹已核对为：

```text
060A 61C5 1B55 8A7F 742B 77AA C52F EB6B 621E 9F35
```

安装包暂存于：

```text
192.168.110.102:/data/xys/slime-coding-agent/docker-rpms-centos7
192.168.110.101:/home/whz/docker-rpms-centos7
```

服务状态：

```bash
systemctl is-enabled docker
systemctl is-active docker
docker version
stat -c '%a %U %G %n' /var/run/docker.sock
```

预期为 `enabled`、`active`、Engine `24.0.9`，socket 权限为 `660 root docker`。
`whz` 已加入 `docker` 组；跨机运行不再需要 root 或 sudo。

## 4. SSH 公钥链路

从 `.102` 验证：

```bash
ssh \
  -i /home/xys/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  -o IdentitiesOnly=yes \
  -o BatchMode=yes \
  -o PasswordAuthentication=no \
  whz@192.168.110.101 'id; hostname; docker version'
```

预期用户为 `whz`、主机名为 `cu01`，并能看到 Docker client/server 24.0.9。
专用公钥指纹：

```text
SHA256:cP7JHiVbXWtEnHlAwwE08dmokeOmdwucRVa0W+lGhhY
```

不要把私钥复制进 runtime image、task image、sandbox container、仓库或日志。

## 5. `.102` 前置资产

```bash
test -f /home/xys/models/Qwen3.5-4B/config.json
test -f /data/xys/slime-coding-agent/assets/node-v22-linux-x64.tar.xz
test -f /data/xys/slime-coding-agent/assets/anthropic-ai-claude-code.tgz
test -f /data/xys/slime-coding-agent/models/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt
docker image inspect slime-coding-agent-runtime:local
docker image inspect slime-coding-agent-swe-smoke:local
```

默认继续使用物理 GPU 1：

```bash
export SLIME_CODING_AGENT_GPU_DEVICE=1
```

## 6. 分层验证与运行

以下命令都在 `.102:/home/xys/slime` 执行。

### 6.1 SSH 和远端 Docker preflight

```bash
cd /home/xys/slime
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh preflight
```

必须看到：

- GPU/runtime 节点为 `master (192.168.110.102)`。
- sandbox 节点为 `whz@192.168.110.101`，主机名 `cu01`。
- Docker server 为 24.0.9 / API 1.43。
- `Remote Docker SSH tunnel: OK`。

### 6.2 同步已验证的 task image

`.101` 不依赖访问 Docker Hub。把 `.102` 上已经由原单机流程验证的 image 通过
`docker save`、SSH tunnel 和 `docker load` 同步过去：

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh build
```

脚本会比较两端完整 RootFS layer digest 列表，不一致时失败。当前本地 task image：

```text
slime-coding-agent-swe-smoke:local
sha256:172e4c1e1e87b2740c4c72fa01a01bd859204f758aec930cd45d88eee2f7a402
```

`docker load` 可能规范化顶层 image config，使 `.101` 显示不同的 image ID；只要
完整 RootFS layer digest 列表一致且后续 smoke 通过，就不表示文件内容发生变化。

这一步不传输 runtime image；`slime-coding-agent-runtime:local` 始终只在 `.102`
运行。

### 6.3 无模型 sandbox smoke

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh sandbox-smoke
```

预期：

```text
DockerSandbox exec/write/read, source diff capture, and clean evaluation passed
```

该步骤实际覆盖跨机 `docker run/exec/cp/rm`、源码 diff、全新 clean-eval container
以及异常清理。

### 6.4 Claude Code 安装 smoke

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh claude-smoke
```

预期包含 `2.1.212 (Claude Code)`。Node 和 Claude Code tarball 来自 `.102`，由
Docker CLI 经 SSH tunnel 复制到 `.101` 的临时容器，不要求共享 `/data`。

### 6.5 Rollout-only

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh rollout
```

运行日志和 dump 仍在 `.102`：

```text
/data/xys/slime-coding-agent/runs/rollout_*/run.log
/data/xys/slime-coding-agent/runs/rollout_*/rollout_dumps/rollout_0.pt
```

成功门槛必须同时满足：

```text
reward=1.00
applied=True
agent_exit_code=0
Ray Job succeeded
```

### 6.6 单步 GRPO

只有跨机 rollout 达到以上四个门槛后再执行：

```bash
bash examples/coding_agent_rl/local_docker/run_remote_docker.sh train
```

checkpoint 仍保存在 `.102` 本次 `RUN_ROOT/checkpoints`，不会写入 `.101`。

### 6.7 2026-07-17 实测通过记录

跨机 rollout 已通过：

```text
RUN_ROOT=/data/xys/slime-coding-agent/runs/rollout_20260717_114256
reward=1.00
applied=True
agent_exit_code=0
response_len=418
Ray Job succeeded
```

跨机单步 GRPO 已通过：

```text
RUN_ROOT=/data/xys/slime-coding-agent/runs/train_20260717_114929
2/2 trajectories: reward=1.00, applied=True, agent_exit_code=0
step=0
global_batch_size=2
lr=1e-6
checkpoint iteration=0
checkpoint size=55G
Ray Job succeeded
```

关键产物：

```text
/data/xys/slime-coding-agent/runs/train_20260717_114929/run.log
/data/xys/slime-coding-agent/runs/train_20260717_114929/rollout_dumps/rollout_0.pt
/data/xys/slime-coding-agent/runs/train_20260717_114929/checkpoints/iter_0000000
/data/xys/slime-coding-agent/runs/train_20260717_114929/checkpoints/latest_checkpointed_iteration.txt
```

`latest_checkpointed_iteration.txt` 的内容为 `0`。训练退出后已确认 `.101` 无残留
`slime-sandbox-*` 容器，`.102` 无残留 runtime container 和临时 tunnel socket，GPU 1
已释放。agent 测试结果为 `65 passed, 1 skipped`。

## 7. 安全与清理

`whz` 的 docker 组权限等价于 `.101` 的 Docker/root 级权限。只运行可信 task
image、harness 资产和受控 rollout；不要把远端 Docker socket 暴露为 TCP，也不要
把 SSH 私钥放进容器。

查看 `.101` 上本流程创建的 sandbox：

```bash
ssh \
  -i /home/xys/.ssh/slime_docker_whz_192_168_110_101_ed25519 \
  whz@192.168.110.101 \
  "docker ps -a --filter label=slime.agent.sandbox=true"
```

正常和异常退出都会删除 task/eval container；外层脚本会删除临时 SSH socket。
不要执行宽泛的 `docker system prune` 或批量 `docker rm`，以免影响 `.101` 的其他
任务。若进程被 `SIGKILL`，只删除确认属于本次运行、名称以 `slime-sandbox-` 开头且
带 `slime.agent.sandbox=true` 标签的容器。

## 8. 常见失败

### SSH tunnel 未创建 socket

运行 `preflight`，检查专用密钥、`known_hosts`、`.101` SSH 服务、Docker 服务和
`whz` 的 docker 组。不要改为 `tcp://0.0.0.0:2375`。

### 远端 image 不存在

执行 `build`，从 `.102` 同步可信 image。生产 dataset 的每个
`metadata.image` 都必须预先存在于 `.101` Docker daemon。

### Sandbox 无法访问 adapter

运行期间从 `.101` 检查：

```bash
curl -v --max-time 3 http://192.168.110.102:18001/
```

任何 HTTP 响应都表示路由存在；超时或拒绝才是端口发布、路由或防火墙问题。

### 回到原单机方案

先让跨机脚本正常退出，使 tunnel 和本地 runtime container 清理完成，然后执行：

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh sandbox-smoke
bash examples/coding_agent_rl/local_docker/run_host.sh rollout
```

原入口继续挂载 `.102:/var/run/docker.sock`，不依赖 `.101`。
