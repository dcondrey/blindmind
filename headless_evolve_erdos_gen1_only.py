"""Rerun of headless_evolve_erdos.py's generation 1, skipping the seed step
(seeds already exist in the db from the first, resource-starved attempt).
That first attempt ran concurrently with an unrelated n=100,000 keiper_li
compute job pinning 350-694% CPU and got zero survivors (19 timeouts, 0
kept) -- almost certainly CPU starvation of the claude CLI subprocess
calls, not a bug. Rerunning now that the machine is free, launched
detached (nohup) since the Bash tool's ~10 minute foreground/background
kill ceiling killed the first same-session retry attempt before it could
produce any output.
"""
import asyncio
import logging

from blindmind.config import settings

settings.variation_temperature = 1.2
settings.critic_temperature = 0.1
settings.critic_threshold = 2.0
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
log = logging.getLogger("erdos_gen1_retry")

PROJECT = "erdos"
POPULATION = 10

DIRECTIVE0 = (
    "Propose concrete computational attack strategies or genuinely applicable cross-domain "
    "techniques for the target puzzles above (EFL conjecture gaps, Egyptian-fraction gap "
    "conjecture, square-packing f(5)=2). HARD RULES: "
    "(1) Any named theorem, algorithm, or technique you invoke must be attributed (name the "
    "method/author, e.g. 'SAT-Modulo-Symmetries', 'branch-and-bound interval arithmetic', "
    "'a specific known reduction technique') or explicitly flagged as your own speculative "
    "proposal, not stated as an established fact you cannot name. "
    "(2) Your mechanism must apply to the ACTUAL mathematical object in the target puzzle -- "
    "do not propose a technique from one seed concept and misapply it to a different one "
    "without checking it actually type-checks (e.g. a coloring technique doesn't "
    "automatically transfer to a packing problem just because both are 'combinatorial'). "
    "(3) You must explicitly address search-space size or computational tractability -- "
    "state (even roughly) how big the relevant search/verification space is for your "
    "proposed approach, and why it's reachable on a single modern workstation (10 cores, "
    "32GB RAM) or a moderate cloud budget (up to a few hundred dollars), not why it's "
    "theoretically interesting. If you cannot estimate this, say so explicitly rather than "
    "omitting it. "
    "(4) Avoid vague physics/biology metaphor dressing (phase transitions, evolutionary "
    "dynamics, ecosystems) unless the target problem has an ACTUAL published connection to "
    "that field you can cite -- these three problems are combinatorics/number-theory/"
    "geometry, not physics, and forcing a physics metaphor onto them without grounding is a "
    "known failure mode."
)


async def main():
    async for session in get_async_session():
        engine = EvolutionEngine(session, directive=DIRECTIVE0, project=PROJECT)

        async def on_survivor(res, session=session):
            await persist_survivor(
                session, PROJECT, 1, res, log, score_fmt=".2f", show_prior_art=False, show_flaws=True
            )

        survivors = await engine.run_generation_cycle(1, POPULATION, on_survivor=on_survivor)
        log.info(f"Gen 1: {len(survivors)} retained")
        break
    s = llm_engine.stats.summary
    log.info(f"LLM: {s['total_calls']} calls, {s.get('failed', 0)} failed, "
             f"{s['input_tokens'] + s['output_tokens']:,} tokens")


if __name__ == "__main__":
    asyncio.run(main())
