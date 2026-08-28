"""Headless BVSR run for project 'riemann' (no human-in-the-loop).

Reuses BlindMind's EvolutionEngine.run_generation_cycle (which already retains by AI-critic
composite score >= threshold) and persists survivors exactly as the CLI does, minus the Stage-2
human prompt. Selection here is critic-only by design, with a deliberately low threshold: I am
the filter, not the critic (same convention as headless_evolve.py for the embodied/conscious
projects).

Uses the claude-cli provider (subscription auth, no per-token billing) since OpenAI/Anthropic/
Gemini API keys on this machine are all out of credit or invalid, and Groq's hardcoded model was
decommissioned.
"""
import asyncio
import logging

from blindmind.config import settings

settings.variation_temperature = 1.1
settings.critic_temperature = 0.1
settings.critic_threshold = 2.0           # capture candidates for human curation; I am the filter, not the critic
settings.crossover_rate = 0.6
settings.point_mutation_rate = 0.3
settings.max_concurrent_calls = 3

from blindmind.engine import EvolutionEngine
from blindmind.db import get_async_session, save_concept
from blindmind.models import Concept
from blindmind.llm import llm_engine, CLAUDE_CLI_LABEL

llm_engine.providers = [{"model": "claude-cli", "api_key": None, "label": CLAUDE_CLI_LABEL}]
llm_engine._initialized = True

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("headless_evolve_riemann")

PROJECT = "riemann"
GENERATIONS = 2
POPULATION = 6
DIRECTIVE0 = (
    "TARGET PUZZLE: on the modular surface PSL(2,Z)\\H, the odd-symmetry class of Maass cusp "
    "forms is fit almost exactly (~0% deviation, residual std 0.7) by the quartic-plus-log local "
    "density basis {T^2, T log T, T, 1}. The even-symmetry class is NOT: three independent "
    "robust-regression methods (targeted outlier exclusion, Huber-loss IRLS blind to outlier "
    "location, multi-region exclusion) all converge on the same ~94-99% upward deviation in the "
    "leading coefficient, ruling out a single removable data anomaly. Real LMFDB data (2202 "
    "level-1 forms) already computed and validated; this is a genuine structural mismatch, not a "
    "bug. Generate concrete, falsifiable hypotheses for WHY the even class's counting function "
    "would have a genuinely different functional SHAPE than the odd class's, given both live on "
    "the same surface and same Laplacian, differing only by behavior under z -> -zbar. Prioritize "
    "hypotheses that (a) point to a SPECIFIC alternative functional form or unfolding method "
    "(not just 'try harder'), and (b) are grounded in real spectral theory, trace-formula, or "
    "arithmetic-quantum-chaos mechanisms -- not vague analogy. A hypothesis is only useful if it "
    "implies a concrete next computation on the existing LMFDB data. Avoid woo (no vague "
    "resonance/numerology). Do NOT merely restate the seeds -- produce sharper, testable, "
    "buildable refinements that a follow-up numerical analysis could actually falsify."
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
                log.info(f"  kept [{mutation.domain}] {mutation.title} "
                         f"(score {critique.composite_score:.1f}, priorArt {critique.prior_art_overlap}/10)")
            if weighted:
                directive = EvolutionEngine.synthesize_directives(weighted)
            log.info(f"Gen {gen}: {len(survivors)} retained")
        break

    s = llm_engine.stats.summary
    log.info(f"LLM: {s['total_calls']} calls, {s.get('failed', 0)} failed, "
             f"{s['input_tokens'] + s['output_tokens']:,} tokens")


if __name__ == "__main__":
    asyncio.run(main())
