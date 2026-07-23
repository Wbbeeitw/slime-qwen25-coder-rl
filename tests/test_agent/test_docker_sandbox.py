"""Unit coverage for the local Docker sandbox backend."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent.sandbox import DockerSandbox, E2BSandbox, create_sandbox


def test_create_sandbox_selects_backend(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_SANDBOX_BACKEND", raising=False)
    assert isinstance(create_sandbox("image"), E2BSandbox)

    monkeypatch.setenv("SLIME_AGENT_SANDBOX_BACKEND", "docker")
    assert isinstance(create_sandbox("image"), DockerSandbox)


def test_docker_sandbox_lifecycle_uses_network_and_cleans_up(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_DOCKER_NETWORK", "agent-net")
    sandbox = DockerSandbox("task-image")
    calls = []

    async def fake_docker(*args, **kwargs):
        calls.append((args, kwargs))
        return 0, "container-id\n", ""

    monkeypatch.setattr(sandbox, "_docker", fake_docker)

    async def run_case():
        async with sandbox:
            assert sandbox._started

    asyncio.run(run_case())

    run_args = calls[0][0]
    assert run_args[0] == "run"
    assert ("--network", "agent-net") == (run_args[run_args.index("--network")], run_args[run_args.index("--network") + 1])
    assert "task-image" in run_args
    assert calls[-1][0][:2] == ("rm", "-f")


def test_docker_sandbox_path_copy_and_chown(monkeypatch, tmp_path: Path):
    source = tmp_path / "asset.tgz"
    source.write_bytes(b"asset")
    sandbox = DockerSandbox("task-image")
    docker_calls = []
    exec_calls = []

    async def fake_docker(*args, **kwargs):
        docker_calls.append((args, kwargs))
        return 0, "", ""

    async def fake_exec(cmd, **kwargs):
        exec_calls.append((cmd, kwargs))
        return 0, "", ""

    monkeypatch.setattr(sandbox, "_docker", fake_docker)
    monkeypatch.setattr(sandbox, "exec", fake_exec)
    asyncio.run(sandbox.write_file("/tmp/asset.tgz", source, user="agent"))

    assert docker_calls[0][0] == (
        "cp",
        "--follow-link",
        str(source),
        f"{sandbox.sandbox_id}:/tmp/asset.tgz",
    )
    assert any("chown agent:agent" in cmd for cmd, _ in exec_calls)
