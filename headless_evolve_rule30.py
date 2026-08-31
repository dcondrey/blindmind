"""Headless BVSR run for project 'rule30' (no human-in-the-loop).

Reuses BlindMind's EvolutionEngine.run_generation_cycle (which already retains by AI-critic
composite score >= threshold) and persists survivors exactly as the CLI does, minus the Stage-2
human prompt. Selection here is critic-only by design, with a deliberately low threshold: I am
the filter, not the critic (same convention as headless_evolve_riemann.py).

Uses the claude-cli provider (subscription auth, no per-token billing).

Separate idea-generation channel from the OpenRouter roundtable panel in
/Volumes/A/researchpapers/13-rule30/experiments/overnight-arms/roundtable/: that script runs a
fixed 4-model panel producing one deep lateral spark per turn; this one runs BVSR's
crossover/mutation loop over a seed population, which explores by recombining concepts rather
than single-model association, so it is a genuinely different generator, not a redundant one.
"""
import asyncio
import logging

from blindmind.config import settings

settings.variation_temperature = 1.2
settings.critic_temperature = 0.1
settings.critic_threshold = 2.0           # capture candidates for human curation; I am the filter, not the critic
settings.crossover_rate = 0.6
settings.point_mutation_rate = 0.3
# Serial, not the usual default: this project's directive is long enough (register
# digest + forbidden-domain enumeration) that concurrent claude-cli subprocesses
# reliably pushed the subprocess timeout in llm.py, timing out every candidate in
# the first live run. A single ~73s call succeeds. LLMEngine now pins claude-cli to
# concurrency=1 unconditionally (its own semaphore, separate from
# max_concurrent_calls, which now only governs real API providers), so this line is
# redundant for that guarantee -- kept anyway as an explicit, self-documenting
# statement of intent for this script.
settings.max_concurrent_calls = 1

from blindmind.db import get_async_session, save_concept
from blindmind.engine import EvolutionEngine
from blindmind.llm import CLAUDE_CLI_LABEL, llm_engine
from blindmind.models import Concept

llm_engine.providers = [{"model": "claude-cli", "api_key": None, "label": CLAUDE_CLI_LABEL}]
llm_engine._initialized = True

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("headless_evolve_rule30")

PROJECT = "rule30"
GENERATIONS = 3
POPULATION = 8
DIRECTIVE0 = (
    "TARGET PROBLEM: Wolfram's Rule 30 elementary cellular automaton, center-column sequence "
    "from a single-cell seed. Three open questions, ranked by tractability: P1 (does the center "
    "column ever become periodic? -- believed no, unproven), P2 (is the center column "
    "equidistributed / normal? -- believed yes, unproven), P3 (is Omega(n) computational "
    "resource genuinely required to determine cell n, i.e. no shortcut algorithm faster than "
    "direct simulation? -- open). The ranking filter for any candidate idea is one question: "
    "does this argument also apply to Rule 90? Rule 90 is left-permutive and additive and its "
    "own single-seed center column is EVENTUALLY PERIODIC (identically zero after t=0) -- so any "
    "argument that would 'prove' periodicity-related properties for Rule 90 too is vacuous and "
    "must be rejected regardless of how compelling it looks for Rule 30. "
    ""
    "TWO PRIOR IDEA-GENERATION PANELS (a 4-model OpenRouter roundtable, formal-logic mode then "
    "lateral-thinking mode) already produced 13+ candidate directions, ALL independently verified "
    "KILLED. Every single one reduced to one of three dead argument shapes: (a) a bounded-strip / "
    "cone-width propagation bound (same content as the O(log t) forcing wall already known), "
    "(b) an ergodic-measure or mutual-information statement that is undefined or trivial on a "
    "single deterministic orbit (the measure-zero single-orbit gap), or (c) a circuit/preimage "
    "counting statement that does not yield a genuine lower bound (same shape as the already-known "
    "counting route). A candidate idea is worthless if, once formalized, it collapses into one of "
    "these three shapes wearing a different vocabulary. "
    ""
    "ALREADY-EXHAUSTED SOURCE DOMAINS (do not reskin these, the panels covered them exhaustively): "
    "stage magic/cons, economics/currency systems, music theory, visual art/optics, literary "
    "branching-time tropes, epistemic logic puzzles, thermodynamics/statistical mechanics as such, "
    "biology's usual suspects (prion misfolding, morphogens, ant colonies, immunology), formal "
    "logic/proof theory, card games/shuffles, dendrochronology, personal-identity philosophy "
    "puzzles, quantum information framings, general relativity, random matrix theory, gauge "
    "theory/holonomy, 2-adic/generating-function rationality arguments, bodily/medical experience "
    "phenomena (phantom limbs, tinnitus, sleep paralysis, addiction cycles, autoimmunity), "
    "obsolete/failing technology (dead media formats, fax/telegraph noise, tape hiss, numbers "
    "stations, CRT interference), culturally specific ritual/belief (glossolalia, seance/ouija, "
    "curse structures, funerary practice), crime/forgery/deception technique (money laundering "
    "layering, art forgery craquelure, gaslighting, smuggling), and linguistic/cultural drift "
    "(rumor mutation, forensic stylometry, dead languages, glitch-art compression). "
    ""
    "Use BVSR's actual strength here, which is different from a single-model spark: CROSS a pair "
    "of the seed concepts below against each other (not against a fresh outside domain -- that is "
    "the roundtable's job) to find a structural combination neither seed states alone, or MUTATE "
    "one seed by tightening it into a specific, formalizable claim about the Rule 30 update map "
    "(a degree-2 GF(2) polynomial map with a specific dependency structure: cell t+1 at position i "
    "= left XOR (center OR right)) that a Rule 90 sanity check could concretely falsify. Every "
    "output must end with an explicit Rule-90 gut check: does this argument's conclusion also "
    "follow for Rule 90's update rule (left XOR center XOR right)? If yes, or if there's nothing "
    "for Rule 90 to check because the object doesn't exist there, say so plainly -- a vacuous pass "
    "is weak evidence, not a false one. Do NOT produce vague inspirational restatements; every "
    "surviving concept must sketch what the Rule-30-specific mathematical object would actually be."
)

SEEDS = [
    ("dynamics", "Non-uniform Bernoulli mixing gap",
     "The center column is conjectured normal/equidistributed (P2), but the map is only "
     "left-permutive, not two-sided permutive like Rule 90's additive structure -- so any measure "
     "argument needs a substitute for the missing algebraic symmetry group. What replaces group "
     "invariance when the only symmetry is one-sided?"),
    ("complexity", "Omega(n) lower bound via incompressibility of the trace",
     "P3 asks whether determining cell n requires simulating ~n steps. A genuine lower bound needs "
     "an argument that no shortcut circuit of subquadratic size can predict the column, without "
     "assuming an unproven circuit lower bound as a black box."),
    ("algebra", "GF(2) polynomial degree growth of the reachable-state map",
     "Rule 30 update is x_i' = x_{i-1} XOR (x_i OR x_i+1) = x_{i-1} XOR x_i XOR x_i+1 XOR x_i x_i+1 "
     "-- the AND term is the entire source of non-linearity and the entire reason Rule 30 differs "
     "from Rule 90's purely linear (XOR-only) update. Any argument that does not use this AND term "
     "explicitly is Rule-90-blind and dies immediately."),
    ("combinatorics", "Preimage tree branching asymmetry",
     "Rule 30 is not injective (multiple predecessor rows map to the same successor row); Rule 90 "
     "restricted to the relevant coset also has structured non-injectivity. The known counting "
     "route failed to turn preimage-count asymmetry into a lower bound -- what invariant of the "
     "preimage TREE's shape (not just its size) survives the Rule-90 comparison?"),
    ("logic", "Forcing/obligation propagation past the O(log t) wall",
     "Known result: determining the center cell at time t forces roughly the outermost O(log t) "
     "bits of the initial row (bounded light-cone argument), and this wall has resisted every "
     "attempt to push further. What kind of dependency would have to exist BETWEEN forced bits "
     "(not just a bigger forced SET) to break past a purely width-based wall?"),
    ("statistics", "Finite-window predicate insufficiency for infinite non-periodicity",
     "A finite window of the column can never, by itself, certify non-periodicity of an infinite "
     "sequence (any finite prefix is consistent with eventual periodicity). What OTHER kind of "
     "finite, checkable object (not a window of values) could certify non-periodicity the way a "
     "continued-fraction tail can certify irrationality?"),
]


async def main():
    directive = DIRECTIVE0
    async for session in get_async_session():
        for concept_domain, title, description in SEEDS:
            seed = Concept(project=PROJECT, domain=concept_domain, title=title,
                            description=description, generation=0, fitness_score=None,
                            tags="seed")
            await save_concept(session, seed, parent_ids=[], mutation_type=None)
        for gen in range(1, GENERATIONS + 1):
            log.info(f"=== Generation {gen} ===")
            engine = EvolutionEngine(session, directive=directive, project=PROJECT)

            saved = []  # (directive, score) pairs, filled in as survivors are found

            async def on_survivor(res, saved=saved):
                mutation, critique, parent_ids, m_type = res
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
                saved.append((critique.evolutionary_directive, critique.composite_score))
                log.info(f"  kept [{mutation.domain}] {mutation.title} "
                         f"(score {critique.composite_score:.1f}, priorArt {critique.prior_art_overlap}/10)")

            try:
                # Outer sanity ceiling, not the real hang bound (that's llm.py's
                # per-call 240s timeout). run_generation_cycle can run up to
                # population_size*5 attempts across multiple batches before
                # returning, so worst case is much larger than one batch; each
                # survivor is saved via on_survivor as soon as it's found, so a
                # timeout here can no longer discard already-completed work.
                await asyncio.wait_for(
                    engine.run_generation_cycle(gen, POPULATION, on_survivor=on_survivor), timeout=5400
                )
            except TimeoutError:
                log.warning(f"Gen {gen} timed out after 5400s; {len(saved)} survivors saved before timeout")

            if saved:
                directive = EvolutionEngine.synthesize_directives(saved)
            log.info(f"Gen {gen}: {len(saved)} retained")
        break

    s = llm_engine.stats.summary
    log.info(f"LLM: {s['total_calls']} calls, {s.get('failed', 0)} failed, "
             f"{s['input_tokens'] + s['output_tokens']:,} tokens")


if __name__ == "__main__":
    asyncio.run(main())
