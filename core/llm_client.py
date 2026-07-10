"""
llm_client.py
--------------
A pluggable LLM client for the multi-agent accelerator planner.

Design goal: the whole system must RUN OUT OF THE BOX with no API key,
so we ship a deterministic MockLLM that produces structured, domain-aware
text. When a real key is available (ANTHROPIC_API_KEY in the environment),
the same .complete() interface routes to the real API instead.

This separation means the orchestration / memory / logging logic is
identical whether you run offline (for grading/demo) or online (for
production) — a testable-by-design pattern: no API cost or flakiness
during development, and deterministic outputs for reproducible demos.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class MockLLM:
    """
    Deterministic, offline stand-in for a real LLM.

    It is NOT random: given the same prompt it returns the same text, which
    keeps the demo reproducible. It inspects lightweight cues in the prompt
    (role + domain) to return role-appropriate, domain-aware content so the
    multi-agent collaboration reads sensibly.
    """

    provider = "mock"
    model = "mock-deterministic-v1"

    # Small built-in domain knowledge so the demo looks credible offline.
    _DOMAIN_NOTES = {
        "fintech": {
            "trends": "embedded finance, open banking APIs, real-time payments, RegTech automation",
            "funding": "Barclays Accelerator, FCA Innovation Hub grants, Seedcamp, Anthemis early-stage",
            "pitch_angle": "regulatory moats and payment-rail integrations",
        },
        "healthtech": {
            "trends": "remote patient monitoring, AI triage, interoperability (FHIR), digital therapeutics",
            "funding": "NHS Clinical Entrepreneur Programme, Wellcome Trust, KHP Ventures, EIT Health",
            "pitch_angle": "clinical validation pathways and NHS procurement readiness",
        },
        "climatetech": {
            "trends": "carbon accounting platforms, grid flexibility, industrial heat decarbonisation, nature-based credits",
            "funding": "Breakthrough Energy Fellows, Innovate UK Net Zero grants, Clean Growth Fund, EIC Accelerator",
            "pitch_angle": "measurable abatement per pound invested",
        },
    }

    _DEFAULT_NOTES = {
        "trends": "AI-driven automation, vertical SaaS specialisation, API-first infrastructure",
        "funding": "generalist pre-seed funds, university accelerators, government innovation grants",
        "pitch_angle": "clear wedge into an underserved workflow",
    }

    def _domain_for(self, prompt: str) -> dict:
        lower = prompt.lower()
        for domain, notes in self._DOMAIN_NOTES.items():
            if domain in lower:
                return {"name": domain, **notes}
        return {"name": "general", **self._DEFAULT_NOTES}

    def complete(self, prompt: str) -> LLMResponse:
        d = self._domain_for(prompt)
        # Route on the instruction (text before the recalled-context block),
        # not the whole prompt — otherwise keywords inside recalled memory
        # (e.g. 'research') would hijack later steps like refine/critique.
        instruction = prompt.split("Prior context:")[0].lower()
        lower = instruction

        if "refine" in lower or "revise" in lower:
            text = textwrap.dedent(f"""\
                Refined pitch outline (incorporating critique):
                1. Problem — quantified pain point with a headline metric.
                2. Solution — one-line wedge, emphasising {d['pitch_angle']}.
                3. Market — trend evidence: {d['trends']}.
                4. Competition — named rivals plus explicit differentiation wedge.
                5. Traction & plan — milestones mapped to the funding ask.
                6. Ask — amount, use-of-funds split, target programmes: {d['funding']}.""")
            return LLMResponse(text=text, provider=self.provider, model=self.model)

        if "critique" in lower or "critic" in lower or "review" in lower:
            text = textwrap.dedent("""\
                Critique of the draft outline:
                1. The problem slide asserts pain without quantifying it — add a number.
                2. The competitive slide lists rivals but not the wedge — state why now, why us.
                3. Funding ask lacks a use-of-funds split — investors will ask immediately.
                Overall: solid skeleton, needs sharper evidence and a clearer ask.""")
            return LLMResponse(text=text, provider=self.provider, model=self.model)

        if "research" in lower and "market" in lower:
            text = textwrap.dedent(f"""\
                Market research — {d['name']}:
                Key current trends: {d['trends']}.
                The segment is seeing increased early-stage activity, with
                differentiation shifting from features to distribution and
                compliance readiness. New entrants should expect a 12-18 month
                window before incumbent responses.""")
        elif "funding" in lower or "grant" in lower:
            text = textwrap.dedent(f"""\
                Funding recommendations — {d['name']}:
                Relevant programmes: {d['funding']}.
                Recommended approach: apply to one flagship accelerator for
                signal value, pair it with one non-dilutive grant, and hold
                institutional pre-seed conversations until traction data exists.""")
        elif "critique" in lower or "review" in lower:
            text = textwrap.dedent("""\
                Critique of the draft outline:
                1. The problem slide asserts pain without quantifying it — add a number.
                2. The competitive slide lists rivals but not the wedge — state why now, why us.
                3. Funding ask lacks a use-of-funds split — investors will ask immediately.
                Overall: solid skeleton, needs sharper evidence and a clearer ask.""")
        elif "refine" in lower or "revise" in lower:
            text = textwrap.dedent(f"""\
                Refined pitch outline (incorporating critique):
                1. Problem — quantified pain point with a headline metric.
                2. Solution — one-line wedge, emphasising {d['pitch_angle']}.
                3. Market — trend evidence: {d['trends']}.
                4. Competition — named rivals plus explicit differentiation wedge.
                5. Traction & plan — milestones mapped to the funding ask.
                6. Ask — amount, use-of-funds split, target programmes: {d['funding']}.""")
        elif "pitch" in lower or "outline" in lower or "deck" in lower:
            text = textwrap.dedent(f"""\
                Draft pitch deck outline — {d['name']}:
                1. Problem — the workflow pain in this segment.
                2. Solution — our approach, angled on {d['pitch_angle']}.
                3. Market — why now: {d['trends']}.
                4. Competition — current landscape overview.
                5. Ask — indicative raise aligned to: {d['funding']}.""")
        else:
            text = f"[mock completion for domain: {d['name']}] {prompt[:120]}"

        return LLMResponse(text=text, provider=self.provider, model=self.model)


class AnthropicLLM:
    """
    Real-API client. Only constructed when ANTHROPIC_API_KEY is set.
    Same .complete() interface as MockLLM, so the rest of the system
    doesn't know or care which one it's talking to.
    """

    provider = "anthropic"
    model = "claude-sonnet-4-6"

    def __init__(self):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package not installed — run: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def complete(self, prompt: str) -> LLMResponse:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if block.type == "text"
        )
        return LLMResponse(text=text, provider=self.provider, model=self.model)


def get_llm():
    """
    Factory: returns AnthropicLLM if a key is configured, else MockLLM.
    The orchestrator and agents only ever call this — they never import
    a concrete client directly.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicLLM()
    return MockLLM()
