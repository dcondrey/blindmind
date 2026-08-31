"""Headless BVSR run for project 'riemann', generation 3.

Direct continuation of headless_evolve_riemann.py's generation 1-2 (see
riemann-attempts.txt attempt #16 for the full write-up of that run and its
curation). Reuses BlindMind's EvolutionEngine.run_generation_cycle exactly
as before -- critic-only retention at a deliberately low threshold, human
(the assistant, this session) is the real filter, not the critic.

WHY A NEW DIRECTIVE: attempt #16's curation found the two highest-scoring
gen1/gen2 lineages both invoked Dirichlet characters/conductors/twist-level
as the mechanism splitting the even Maass symmetry class -- a category
error, since the actual target dataset is level 1 (PSL(2,Z)), which has
ONLY the trivial character (strong multiplicity one applies; there is
nothing to stratify by). A separate gen2 concept ("Persistent Spectral
Homology") stated an uncited RH-equivalence formula that reads as
fabricated. This directive explicitly forbids both failure modes and
broadens scope back to genuine divergence (blindmind's actual strength,
per project convention) across ALL 11 open threads, not just the one
puzzle -- a parallel, separately-dispatched primary-source research task
is already handling the correctly-scoped redirect of the character-twist
idea (to a real congruence subgroup Gamma_0(N), where nontrivial
characters actually exist); this run is for genuinely new angles, not a
third pass at the same narrow question.
"""

import asyncio
import logging

from blindmind.config import settings

settings.variation_temperature = 1.2
settings.critic_temperature = 0.1
settings.critic_threshold = 2.0  # capture candidates for human curation; I am the filter, not the critic
settings.crossover_rate = 0.6
settings.point_mutation_rate = 0.3
settings.max_concurrent_calls = 3

from blindmind.db import get_async_session  # noqa: E402 (settings must be set before these imports)
from blindmind.engine import EvolutionEngine  # noqa: E402 (settings must be set before these imports)
from blindmind.llm import CLAUDE_CLI_LABEL, llm_engine  # noqa: E402 (settings must be set before these imports)
from headless_common import persist_survivor  # noqa: E402 (settings must be set before these imports)

llm_engine.providers = [{"model": "claude-cli", "api_key": None, "label": CLAUDE_CLI_LABEL}]
llm_engine._initialized = True

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("headless_evolve_riemann_gen3")

PROJECT = "riemann"
GENERATIONS = 1
POPULATION = 8
DIRECTIVE0 = (
    "Generate genuinely novel, cross-domain hypotheses for open Riemann-Hypothesis-adjacent "
    "research threads. HARD RULES, both violated by the previous generation's top-scoring "
    "concepts: "
    "(1) Do NOT invoke Dirichlet characters, conductors, nebentypus, or twist-level as a "
    "mechanism for anything at PSL(2,Z) / level 1 -- level 1 has ONLY the trivial character "
    "(strong multiplicity one), so 'stratify by character' is a category error there. If your "
    "mechanism genuinely needs nontrivial characters, say explicitly that it requires moving to "
    "a congruence subgroup Gamma_0(N) for some N>1, and name a plausible small N. "
    "(2) Do NOT state any named theorem, formula, or 'RH is equivalent to X' claim unless you "
    "can attribute it (author, approximate year, or the paper it comes from). If you cannot "
    "attribute it, say explicitly that it is speculative/unverified -- do not present an "
    "invented formula as an established fact. "
    "(3) Avoid vague biological/ecological metaphor recombination (predator-prey, evolutionary "
    "attractors, Lotka-Volterra) -- already tried, scored low, produced nothing usable. "
    "Prioritize concepts that imply a CONCRETE next computation (a specific dataset, formula, or "
    "statistical test), grounded in real published research programs, over purely conceptual or "
    "philosophical juxtapositions."
)


async def main():
    directive = DIRECTIVE0
    async for session in get_async_session():
        for gen in range(3, 3 + GENERATIONS):
            log.info(f"=== Generation {gen} ===")
            engine = EvolutionEngine(session, directive=directive, project=PROJECT)
            weighted = []

            async def on_survivor(res, gen=gen, weighted=weighted, session=session):
                weighted.append(
                    await persist_survivor(session, PROJECT, gen, res, log, score_fmt=".2f", show_flaws=True)
                )

            survivors = await engine.run_generation_cycle(gen, POPULATION, on_survivor=on_survivor)
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
