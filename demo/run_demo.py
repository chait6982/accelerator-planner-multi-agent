"""
run_demo.py
-----------
CLI entry point for the Accelerator Planner.

Usage:
    python demo/run_demo.py --domain fintech
    python demo/run_demo.py --domain healthtech
    python demo/run_demo.py --domain climatetech

Runs entirely offline by default (deterministic MockLLM). If
ANTHROPIC_API_KEY is set in the environment, the same pipeline runs
against the live API instead — no code changes needed.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running from the repo root or from demo/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import AcceleratorPlanner


def main():
    parser = argparse.ArgumentParser(description="Run the Accelerator Planner demo")
    parser.add_argument(
        "--domain",
        default="fintech",
        help="Startup domain to plan for (e.g. fintech, healthtech, climatetech)",
    )
    args = parser.parse_args()

    planner = AcceleratorPlanner()
    final_outline = planner.run(args.domain)

    print("\n" + "=" * 60)
    print("FINAL REFINED PITCH OUTLINE")
    print("=" * 60)
    print(final_outline)
    print("\nFull reasoning trace saved under the runs/ directory.")


if __name__ == "__main__":
    main()
