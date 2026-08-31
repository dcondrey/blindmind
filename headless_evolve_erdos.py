"""Headless BVSR run for a NEW project 'erdos', seeded with the candidates
found by this session's survey of erdosproblems.com (see the conversation
transcript for the full research write-up; not yet logged to a dedicated
attempts file the way the RH project is -- this is a fresh exploration).

Goal: not to pick a problem, but to diverge on ATTACK STRATEGIES for the
already-identified candidates -- cross-domain techniques that could extend
the documented state of the art -- using blindmind for what it's actually
good at (divergence across distant domains) rather than refining a single
idea. Same "I am the filter, not the critic" convention as the RH project's
headless scripts: threshold low, human curation happens after.

Directive is deliberately restrictive about the SAME failure modes already
found and fixed this session on the RH project's blindmind runs:
uncited/fabricated formulas, category errors (proposing a mechanism that
doesn't apply to the actual object in question), and vague metaphor
recombination with no concrete next step.
"""
import asyncio
import logging

from blindmind.config import settings

settings.variation_temperature = 1.2
settings.critic_temperature = 0.1
settings.critic_threshold = 2.0   # capture candidates for human curation; I am the filter, not the critic
settings.crossover_rate = 0.6
settings.point_mutation_rate = 0.3
settings.max_concurrent_calls = 3

from blindmind.db import get_async_session, save_concept
from blindmind.engine import EvolutionEngine
from blindmind.llm import CLAUDE_CLI_LABEL, llm_engine
from blindmind.models import Concept
from headless_common import persist_survivor

llm_engine.providers = [{"model": "claude-cli", "api_key": None, "label": CLAUDE_CLI_LABEL}]
llm_engine._initialized = True

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("headless_evolve_erdos")

PROJECT = "erdos"
GENERATIONS = 1
POPULATION = 10

SEEDS = [
    {
        "domain": "Combinatorics / Hypergraph coloring",
        "title": "Erdos-Faber-Lovasz conjecture (EFL)",
        "description": (
            "Every linear hypergraph on n vertices is n-edge-colorable. Proven for all "
            "sufficiently large n (Kang-Kelly-Kuhn-Methuku-Osthus, Annals of Math 2023, "
            "arXiv:2101.04698). Small cases verified by an exact SAT-Modulo-Symmetries + "
            "DRAT-certified toolchain (Kirchweger-Peitl-Szeider, SAT 2023): full coverage "
            "n<=12, but real GAPS remain at n=13..18 (e.g. n=13 has an unverified edge-count "
            "range m in (32,55) between two verified sub-ranges), and n>=19 is completely "
            "untouched computationally. Target puzzle: find a technique that could close "
            "one of these specific gaps, or extend past n=18, using exact/certified methods "
            "(SAT solving, DRAT proof certificates, or equivalent) -- not floating point."
        ),
    },
    {
        "domain": "Number theory / Egyptian fractions",
        "title": "Egyptian-fraction consecutive-gap conjecture",
        "description": (
            "For distinct integers 1<n_1<...<n_k (k>=2) with sum of 1/n_i = 1, is "
            "max(n_{i+1}-n_i) always >= 3? The weaker gap>=2 version was proven by Erdos "
            "himself in 1932. Optimality witness: 1 = 1/2+1/3+1/6 achieves gap exactly 3. "
            "This is pure exact-rational arithmetic -- falsifying it means exhibiting one "
            "unit-fraction decomposition of 1 where every consecutive gap is <=2. No "
            "dedicated computational search of this specific question was found in the "
            "literature. Target puzzle: bound how large n_1 and k could possibly be in a "
            "counterexample (the reciprocal-sum budget should constrain this tightly), then "
            "propose a concrete, exact-arithmetic (rational, not floating point) search "
            "strategy over that bounded space."
        ),
    },
    {
        "domain": "Discrete geometry / Packing",
        "title": "Erdos square-packing conjecture (f(5)=2?)",
        "description": (
            "Pack n squares (no common interior point) inside the unit square; f(n) = max "
            "possible sum of side lengths. Conjecture: f(k^2+1)=k. f(k^2)=k is trivial "
            "(Cauchy-Schwarz); Erdos proved f(2)=1 himself. The smallest genuinely open "
            "unrestricted case is f(5)=2 -- no dedicated computational attack on this "
            "specific case was found. This is a finite-dimensional continuous global "
            "optimization / certified-packing problem: 5 squares, free position + size + "
            "rotation, in the style of small-scale Kepler-conjecture computer proofs using "
            "certified interval/ball arithmetic. Target puzzle: either exhibit a certified "
            "packing achieving side-sum > 2 (falsifies it), or propose a rigorous "
            "branch-and-bound strategy over the packing's parameter space that could "
            "certify no such packing exists up to a given resolution."
        ),
    },
    {
        "domain": "Graph theory / Cycle lengths (context: DISQUALIFIED, seeded as a negative example)",
        "title": "Erdos-Gyarfas conjecture -- why this was ruled out",
        "description": (
            "Every graph with minimum degree >=3 has a cycle whose length is a power of 2. "
            "Live, active research (a paper dated Aug 2026 pushed a bipartite-cubic special "
            "case's counterexample-search bound from 30 to 60 vertices), BUT that progress "
            "relied on a structural trick (bipartiteness + a Moore-bound argument collapsing "
            "the search to a small configuration-enumeration problem) specific to that "
            "subcase and already used to its stated limit. The GENERAL (non-bipartite) cubic "
            "case's search space at the next unverified size is ~1.9x10^13 graphs (OEIS "
            "A002851) with no published structural reduction -- not reachable on modest "
            "hardware without new mathematics, not just more compute. SEEDED HERE AS A "
            "CALIBRATION EXAMPLE: any proposed mechanism must explain why it does NOT hit "
            "this same wall (search space too large, no structural reduction available)."
        ),
    },
]

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
    directive = DIRECTIVE0
    async for session in get_async_session():
        seed_concepts = []
        for s in SEEDS:
            c = Concept(project=PROJECT, domain=s["domain"], title=s["title"],
                        description=s["description"], generation=0)
            await save_concept(session, c)
            seed_concepts.append(c)
        log.info(f"Seeded {len(seed_concepts)} concepts into project '{PROJECT}'")

        for gen in range(1, 1 + GENERATIONS):
            log.info(f"=== Generation {gen} ===")
            engine = EvolutionEngine(session, directive=directive, project=PROJECT)
            weighted = []

            async def on_survivor(res, gen=gen, weighted=weighted):
                weighted.append(await persist_survivor(session, PROJECT, gen, res, log, score_fmt=".2f", show_flaws=True))

            survivors = await engine.run_generation_cycle(gen, POPULATION, on_survivor=on_survivor)
            if weighted:
                directive = EvolutionEngine.synthesize_directives(weighted)
            log.info(f"Gen {gen}: {len(survivors)} retained")
        break

    s = llm_engine.stats.summary
    log.info(f"LLM: {s['total_calls']} calls, {s.get('failed', 0)} failed, "
             f"{s['input_tokens'] + s['output_tokens']:,} tokens")


if __name__ == "__main__":
    asyncio.run(main())
