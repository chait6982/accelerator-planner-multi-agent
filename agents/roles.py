"""
roles.py
--------
The four role agents: Research, Funding, Pitch Coach, and Review Critic.

Each agent follows the same pattern:
    1. recall() relevant context from SharedMemory (context handoff in)
    2. build a role-specific prompt around that context
    3. call the pluggable LLM
    4. add() its output back into SharedMemory (context handoff out)
    5. record the whole step with the Tracer

The agents never talk to each other directly — everything flows through
the shared vector memory, which is what makes the handoffs auditable.
"""

from __future__ import annotations

from core.llm_client import get_llm
from core.memory import SharedMemory
from core.tracer import Tracer


class BaseAgent:
    name = "base"
    action = "act"

    def __init__(self, memory: SharedMemory, tracer: Tracer):
        self.memory = memory
        self.tracer = tracer
        self.llm = get_llm()

    def _recall_context(self, query: str) -> list[str]:
        records = self.memory.recall(query, k=3, exclude_author=self.name)
        return [f"({r.author}) {r.content}" for r in records]

    def _run_step(self, query: str, prompt_template: str) -> str:
        context = self._recall_context(query)
        context_block = "\n\n".join(context) if context else "(no prior context)"
        prompt = prompt_template.format(context=context_block)

        response = self.llm.complete(prompt)

        self.memory.add(author=self.name, content=response.text)
        self.tracer.record_step(
            agent=self.name,
            action=self.action,
            recalled_context=context,
            prompt=prompt,
            output=response.text,
            provider=response.provider,
            model=response.model,
        )
        return response.text


class ResearchAgent(BaseAgent):
    name = "ResearchAgent"
    action = "market_research"

    def run(self, domain: str) -> str:
        return self._run_step(
            query=f"market research trends {domain}",
            prompt_template=(
                f"You are a startup market researcher. Research the market for "
                f"the {domain} domain: key trends, competitive dynamics, and "
                f"timing signals.\n\nPrior context:\n{{context}}"
            ),
        )


class FundingAdvisor(BaseAgent):
    name = "FundingAdvisor"
    action = "funding_recommendations"

    def run(self, domain: str) -> str:
        return self._run_step(
            query=f"funding grants accelerator {domain}",
            prompt_template=(
                f"You are a startup funding advisor. Based on the market "
                f"research below, recommend funding programmes and grants for "
                f"an early-stage {domain} startup.\n\nPrior context:\n{{context}}"
            ),
        )


class PitchCoach(BaseAgent):
    name = "PitchCoach"
    action = "draft_pitch_outline"

    def draft(self, domain: str) -> str:
        return self._run_step(
            query=f"pitch deck outline {domain} research funding",
            prompt_template=(
                f"You are a pitch coach. Draft a pitch deck outline for an "
                f"early-stage {domain} startup, synthesising the research and "
                f"funding context below.\n\nPrior context:\n{{context}}"
            ),
        )

    def refine(self, domain: str) -> str:
        # Separate action label so the trace distinguishes draft from refine.
        original_action = self.action
        self.action = "refine_pitch_outline"
        try:
            return self._run_step(
                query=f"critique review pitch outline {domain}",
                prompt_template=(
                    f"You are a pitch coach. Refine and revise the pitch deck "
                    f"outline for the {domain} startup using the critique "
                    f"below. Address every point raised.\n\nPrior context:\n{{context}}"
                ),
            )
        finally:
            self.action = original_action


class ReviewCritic(BaseAgent):
    name = "ReviewCritic"
    action = "critique_pitch"

    def run(self, domain: str) -> str:
        return self._run_step(
            query=f"pitch deck outline {domain}",
            prompt_template=(
                f"You are a tough startup review critic. Critique the pitch "
                f"outline below for a {domain} startup: identify weaknesses, "
                f"missing evidence, and unclear asks.\n\nPrior context:\n{{context}}"
            ),
        )
