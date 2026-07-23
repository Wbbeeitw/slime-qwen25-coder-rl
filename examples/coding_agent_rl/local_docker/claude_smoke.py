"""Install the real Claude Code harness in a local task container."""

from __future__ import annotations

import asyncio

from slime.agent.harness import ClaudeCodeHarness
from slime.agent.sandbox import create_sandbox


async def main() -> None:
    async with create_sandbox("slime-coding-agent-swe-smoke:local") as sandbox:
        try:
            await ClaudeCodeHarness().install_cli(sandbox)
        except Exception:
            _, diagnostics, _ = await sandbox.exec(
                "ls -ld /root /root/.npm /usr/local /usr/local/bin /usr/local/lib 2>&1; "
                "find /root/.npm/_logs -type f -maxdepth 1 -print -exec tail -n 120 {} \\; 2>&1",
                timeout=60,
            )
            print(diagnostics)
            raise
        exit_code, output, stderr = await sandbox.exec("claude --version", timeout=60)
        if exit_code != 0:
            raise RuntimeError(f"claude --version failed: stdout={output!r} stderr={stderr!r}")
        print(output.strip())


if __name__ == "__main__":
    asyncio.run(main())
