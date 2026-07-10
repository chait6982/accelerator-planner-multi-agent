"""
tracer.py
---------
Structured local logging of every step in a run: which agent acted, what
context it recalled from shared memory, what prompt it sent, and what it
produced.

Two artifacts per run:
    run.log     — human-readable, line-per-event log
    trace.json  — machine-readable full trace (list of step records)

This is the "full reasoning traceability" requirement: after a run you can
reconstruct exactly which context each agent used and what it did with it.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class Tracer:
    def __init__(self, out_dir: str = "runs"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(out_dir) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._log_path = self.run_dir / "run.log"
        self._trace_path = self.run_dir / "trace.json"
        self._steps: list[dict] = []
        self._start = time.monotonic()

        self.log(f"Run started at {stamp} UTC")

    def log(self, message: str) -> None:
        elapsed = round(time.monotonic() - self._start, 2)
        line = f"[{elapsed:>7.2f}s] {message}"
        print(line)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def record_step(
        self,
        agent: str,
        action: str,
        recalled_context: list[str],
        prompt: str,
        output: str,
        provider: str,
        model: str,
    ) -> None:
        self._steps.append({
            "step": len(self._steps) + 1,
            "agent": agent,
            "action": action,
            "recalled_context": recalled_context,
            "prompt": prompt,
            "output": output,
            "llm_provider": provider,
            "llm_model": model,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.log(f"{agent} completed '{action}' (context items recalled: {len(recalled_context)})")

    def finalize(self) -> None:
        with open(self._trace_path, "w", encoding="utf-8") as f:
            json.dump(self._steps, f, indent=2)
        self.log(f"Trace written to {self._trace_path}")
        self.log(f"Run complete — {len(self._steps)} steps recorded")
