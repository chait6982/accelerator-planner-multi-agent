"""
orchestrator.py
---------------
Coordinates the agent team in a sequential + feedback-driven flow:

    Research -> Funding -> Pitch (draft) -> Critic (critique) -> Pitch (refine)

The Critic -> refine step is the feedback loop: the PitchCoach's second
pass explicitly consumes the ReviewCritic's output (recalled via shared
memory) and revises the outline in response. That's genuine agentic
feedback, not just a linear pipeline.
"""

from __future__ import annotations

from agents.roles import ResearchAgent, FundingAdvisor, PitchCoach, ReviewCritic
from core.memory import SharedMemory
from core.tracer import Tracer


class AcceleratorPlanner:
    def __init__(self, out_dir: str = "runs"):
        self.memory = SharedMemory()
        self.tracer = Tracer(out_dir=out_dir)

        self.research = ResearchAgent(self.memory, self.tracer)
        self.funding = FundingAdvisor(self.memory, self.tracer)
        self.pitch = PitchCoach(self.memory, self.tracer)
        self.critic = ReviewCritic(self.memory, self.tracer)

    def run(self, domain: str) -> str:
        self.tracer.log(f"Planning run for domain: {domain}")

        self.tracer.log("Step 1/5: market research")
        self.research.run(domain)

        self.tracer.log("Step 2/5: funding recommendations")
        self.funding.run(domain)

        self.tracer.log("Step 3/5: draft pitch outline")
        self.pitch.draft(domain)

        self.tracer.log("Step 4/5: critique (feedback loop in)")
        self.critic.run(domain)

        self.tracer.log("Step 5/5: refine outline (feedback loop out)")
        final_outline = self.pitch.refine(domain)

        self.tracer.finalize()
        return final_outline
