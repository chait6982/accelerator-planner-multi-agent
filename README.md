# Accelerator Planner — Multi-Agent System with Vector Memory & Feedback Loop

A multi-agent workflow planner for a startup accelerator: give it a domain
(e.g. `fintech`, `healthtech`, `climatetech`) and a four-agent team
researches the market, recommends funding, drafts a pitch outline,
critiques it, and refines it — emitting the final outline plus a complete,
auditable reasoning trace.

**Runs entirely offline out of the box** — no API key needed (see
"Pluggable LLM design" below).

> **Note on provenance:** this repo is a rebuild of the original project
> from its documented architecture and recovered code fragments — the
> `SharedMemory` vector store and `MockLLM` design are restored from the
> original build; the orchestrator, agents, and tracer are re-implemented
> to the same documented spec.

## Pipeline

```
Research → Funding → Pitch (draft) → Critic (critique) → Pitch (refine)
```

The **Critic → refine** step is a genuine feedback loop: the PitchCoach's
second pass recalls the ReviewCritic's critique from shared memory and
revises the outline in response — not just a linear pipeline.

## Architecture

| Module | Responsibility |
|---|---|
| `core/llm_client.py` | Pluggable LLM. `MockLLM` (deterministic, offline) by default; `AnthropicLLM` when `ANTHROPIC_API_KEY` is set. |
| `core/memory.py` | `SharedMemory` — a dependency-free **vector store** (TF embedding + cosine similarity) used for context handoffs. |
| `core/tracer.py` | Structured logging of every step: recalled context, prompt, output → `run.log` + `trace.json`. |
| `core/orchestrator.py` | Coordinates the agents in a sequential + feedback-driven flow. |
| `agents/roles.py` | The four role agents (Research, Funding, Pitch Coach, Review Critic). |
| `demo/run_demo.py` | CLI entry point. |

Agents never talk to each other directly — everything flows through the
shared vector memory (`add()` on write, `recall()` by cosine similarity on
read), which is what makes context handoffs auditable in the trace.

## Pluggable LLM design

The whole system runs with **zero API keys** by default: `MockLLM` is a
deterministic, domain-aware stand-in that makes the demo reproducible and
free to run. When `ANTHROPIC_API_KEY` is present in the environment, the
factory returns a real API client with the same `.complete()` interface —
the orchestration, memory, and tracing code is identical either way.

Why this matters: it's a testable-by-design pattern. No API cost or
flakiness during development, deterministic outputs for demos and CI, and
a one-environment-variable switch to production behaviour.

## Framework choice: why a direct implementation (vs CrewAI / LangGraph / AutoGen)

This system deliberately implements the multi-agent pattern **directly**
rather than importing a framework. The trade-offs considered:

- **CrewAI** — excellent for role/task pipelines, but its abstractions hide
  the memory mechanics; this project's goal was to make the vector-memory
  handoff *visible and auditable*, which meant owning that layer.
- **LangGraph** — the right tool for complex, branching stateful graphs;
  this flow is a five-step sequence with one feedback edge, so a full graph
  runtime added weight without adding capability.
- **AutoGen** — strongest for emergent multi-turn agent *conversation*;
  here the coordination is deliberately structured (fixed pipeline + one
  critique loop), so conversation-driven control flow wasn't the fit.

A direct implementation kept the system dependency-free, made every context
handoff explicit in the trace, and — honestly — demonstrated the underlying
pattern rather than a framework's wrapper around it. The CrewAI-style
role/task structure is intentionally preserved so porting to CrewAI later
is straightforward.

## Run it

No install needed for the offline demo (pure standard library):

```bash
python demo/run_demo.py --domain fintech
python demo/run_demo.py --domain healthtech
python demo/run_demo.py --domain climatetech
```

To run against the live Anthropic API instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here   # Windows: set ANTHROPIC_API_KEY=...
python demo/run_demo.py --domain fintech
```

## Reasoning traceability

Every run writes a timestamped folder under `runs/` containing:

- **`run.log`** — human-readable event log (which agent acted, how many
  context items it recalled)
- **`trace.json`** — full machine-readable trace: per step, the exact
  recalled context, the prompt sent, the output produced, and which LLM
  provider/model handled it

This means you can reconstruct, after the fact, exactly which prior agent
outputs influenced each decision — including verifying that the refine step
actually consumed the critique.

## Notes

- No API keys are hardcoded anywhere; the live-API path reads only from the
  environment.
- `runs/` output is gitignored — traces are per-run artifacts, not source.
