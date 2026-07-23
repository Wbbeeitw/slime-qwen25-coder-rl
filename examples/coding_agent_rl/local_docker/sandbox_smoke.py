"""Exercise the real local Docker sandbox contract without a model."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl import swe
from slime.agent.sandbox import create_sandbox, ensure_agent_user


async def main() -> None:
    async with create_sandbox("slime-coding-agent-swe-smoke:local") as sandbox:
        exit_code, output, stderr = await sandbox.exec(
            "cd /workspace/demo && git status --short && pytest -q",
            timeout=120,
        )
        if exit_code == 0:
            raise RuntimeError("the intentionally buggy smoke repository unexpectedly passed")
        if "failed" not in output.lower() and "failed" not in stderr.lower():
            raise RuntimeError(f"pytest failure was not reported: stdout={output!r} stderr={stderr!r}")
        await sandbox.write_file("/tmp/roundtrip.txt", "docker sandbox roundtrip\n")
        assert await sandbox.read_file("/tmp/roundtrip.txt") == "docker sandbox roundtrip\n"
        await ensure_agent_user(sandbox, "/workspace/demo")
        await sandbox.write_file(
            "/workspace/demo/calculator.py",
            "def add(left: int, right: int) -> int:\n    return left + right\n",
            user="agent",
        )
        fixed_ec, _, fixed_err = await sandbox.exec(
            "cd /workspace/demo && pytest -q",
            user="agent",
            timeout=120,
        )
        if fixed_ec != 0:
            raise RuntimeError(f"fixed smoke repository failed: {fixed_err!r}")
        diff_text = await swe.git_diff(sandbox, "/workspace/demo")

    if "calculator.py" not in diff_text or "__pycache__" in diff_text or ".pytest_cache" in diff_text:
        raise RuntimeError(f"captured an invalid source diff: {diff_text!r}")
    result = await swe.run_evaluation(
        {
            "protocol": swe.PROTOCOL_SCALESWE,
            "image": "slime-coding-agent-swe-smoke:local",
            "workdir": "/workspace/demo",
            "grading": {"eval_cmd": "pytest -q"},
        },
        diff_text=diff_text,
        timeout_sec=120,
    )
    if result != swe.EvalResult(1.0, True):
        raise RuntimeError(f"clean evaluation failed: {result!r}")
    print("DockerSandbox exec/write/read, source diff capture, and clean evaluation passed")


if __name__ == "__main__":
    asyncio.run(main())
