# Local Docker smoke run

This setup runs the coding-agent example on one local GPU. The slime runtime is
one GPU-enabled container; each agent attempt and clean evaluation is a sibling
container created through the mounted host Docker socket.

The host entrypoint supports these stages:

```bash
bash examples/coding_agent_rl/local_docker/run_host.sh test
bash examples/coding_agent_rl/local_docker/run_host.sh sandbox-smoke
bash examples/coding_agent_rl/local_docker/run_host.sh claude-smoke
bash examples/coding_agent_rl/local_docker/run_host.sh convert
bash examples/coding_agent_rl/local_docker/run_host.sh rollout
bash examples/coding_agent_rl/local_docker/run_host.sh train
```

Required local images and assets:

- `slime-coding-agent-runtime:local`
- `slime-coding-agent-swe-smoke:local`
- `/data/xys/slime-coding-agent/assets/node-v22-linux-x64.tar.xz`
- `/data/xys/slime-coding-agent/assets/anthropic-ai-claude-code.tgz`

The runtime container mounts `/var/run/docker.sock`. Treat it as host-root
access and do not run untrusted code or untrusted task images with this setup.
