"""Headless BVSR run for project 'conscious' (no human-in-the-loop).

Reuses BlindMind's EvolutionEngine.run_generation_cycle (which already retains by AI-critic
composite score >= threshold) and persists survivors exactly as the CLI does, minus the Stage-2
human prompt. Selection here is critic-only by design; real filtering is done afterward by hand
+ the deep-research literature check.
"""

import asyncio
import logging
import os

from blindmind.config import settings

settings.variation_temperature = 1.3  # divergent
settings.critic_temperature = 0.1
settings.critic_threshold = 2.0  # capture candidates for human curation; I am the filter, not the critic
settings.crossover_rate = 0.7  # bias toward smashing disparate seeds together
settings.point_mutation_rate = 0.2
settings.max_concurrent_calls = 4

from blindmind.db import get_async_session, save_concept  # noqa: E402 (settings must be set before these imports)
from blindmind.engine import EvolutionEngine  # noqa: E402 (settings must be set before these imports)
from blindmind.llm import llm_engine  # noqa: E402 (settings must be set before these imports)
from blindmind.models import Concept  # noqa: E402 (settings must be set before these imports)

llm_engine.providers = [
    {"model": "openrouter/openai/gpt-4o", "api_key": os.environ["OPENROUTER_API_KEY"], "label": "OR-gpt4o"}
]
llm_engine._initialized = True

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("headless_evolve")

PROJECT = "embodied"
GENERATIONS = 2
POPULATION = 8
DIRECTIVE0 = (
    "REFINE and SHARPEN the vision of a continuously-running agent embodied in its REAL compute "
    "substrate: live telemetry (CPU temp, power, latency, memory pressure) as interoceptive senses; "
    "the felt cost of its OWN computation (thinking literally heats the chip); a hardware-grounded "
    "homeostatic affect signal; real irreversible mortality (thermal damage); persistent "
    "continuously-evolving state (not a while-loop) with its own agenda. Generate concrete VARIANTS, "
    "EXTENSIONS, and especially SHARPER EXPERIMENTAL DESIGNS. Top priority: a discriminating "
    "experiment where the REAL felt-cost-of-cognition makes the agent behave in a way a hand-tuned "
    "ABSTRACT cost term provably CANNOT reproduce (otherwise the grounding is decoration). Also "
    "address: extracting a clean affect signal from noisy shared telemetry; what concrete behaviors "
    "to measure (effort rationing, mental fatigue, forced rest). Be concrete and buildable on a real "
    "machine. Avoid woo (no vague resonance/quantum). Do NOT merely restate the seeds - produce "
    "sharper, testable, buildable refinements."
)


async def main():
    directive = DIRECTIVE0
    async for session in get_async_session():
        for gen in range(1, GENERATIONS + 1):
            log.info(f"=== Generation {gen} ===")
            engine = EvolutionEngine(session, directive=directive, project=PROJECT)
            survivors = await engine.run_generation_cycle(gen, POPULATION)
            weighted = []
            for mutation, critique, parent_ids, m_type in survivors:
                mt = getattr(m_type, "value", str(m_type))
                tags = f"gen{gen},{mt},priorart{critique.prior_art_overlap},nov{critique.conceptual_novelty}"
                concept = Concept(
                    project=PROJECT,
                    domain=mutation.domain,
                    title=mutation.title,
                    description=mutation.description,
                    generation=gen,
                    fitness_score=critique.composite_score,
                    tags=tags,
                )
                await save_concept(session, concept, parent_ids=parent_ids, mutation_type=m_type)
                weighted.append((critique.evolutionary_directive, critique.composite_score))
                log.info(
                    f"  kept [{mutation.domain}] {mutation.title} "
                    f"(score {critique.composite_score:.1f}, priorArt {critique.prior_art_overlap}/10)"
                )
            if weighted:
                directive = EvolutionEngine.synthesize_directives(weighted)
            log.info(f"Gen {gen}: {len(survivors)} retained")
        break

    s = llm_engine.stats.summary
    log.info(
        f"LLM: {s['total_calls']} calls, {s.get('failed', 0)} failed, {s['input_tokens'] + s['output_tokens']:,} tokens"
    )


if __name__ == "__main__":
    asyncio.run(main())
