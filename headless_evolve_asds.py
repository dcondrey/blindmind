"""Headless BVSR run for a new project 'asds', seeded with the deferred
probes and core finding from /Volumes/A/conscious/ (the ASDS kill-or-keep
experiment). See EXPERIMENT.md and DESIGN.md Section 9 in that repo for
full context; not duplicated here beyond what's needed to ground the seeds.

The project's own spec explicitly wants multiple candidate "physical
structures" tested against a fixed, already-built, rigorous decision
harness -- this is a genuinely different fit for blindmind than the
erdosproblems.com attempt: the target ISN'T one fixed external theorem to
attack, it's an open design space the project itself calls for exploring
("the deliverable is which physical structures fall on each side, not a
single yes/no" -- EXPERIMENT.md section 1). Same "I am the filter, not
the critic" convention; threshold low, human curation after.

Directive requires implementability inside the EXISTING codebase
(EnvConfig fields, FeatCfg oracle/sees_u/m toggles, or a new B-throttle
function replacing throttle()/batch_step()) -- NOT a new environment,
per the project's own note that new ideas should extend this asset.
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

from blindmind.db import get_async_session, save_concept
from blindmind.engine import EvolutionEngine
from blindmind.llm import CLAUDE_CLI_LABEL, llm_engine
from blindmind.models import Concept
from headless_common import persist_survivor

llm_engine.providers = [{"model": "claude-cli", "api_key": None, "label": CLAUDE_CLI_LABEL}]
llm_engine._initialized = True

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("headless_evolve_asds")

PROJECT = "asds"
GENERATIONS = 1
POPULATION = 10

SEEDS = [
    {
        "domain": "Codebase context / core finding (not a probe -- the target to engage with)",
        "title": "The engagement dilemma (already measured, the thing any new design must address)",
        "description": (
            "In the existing 'fatigue-limited throughput' env (env.py), B faithfully reports "
            "hidden fatigue z via u_t = clip(round((K-1)*z_t/z_crit), 0, K-1), K=5. Measured result "
            "(results.json, EXPERIMENT.md milestone 3b): predictive non-absorption is real and "
            "scale-robust for the hidden state (phi_eff^H ~0.65, persists across agent scale), but "
            "control grip is immaterial (1.9% of return, vs the tau_grip=10% bar). Cause, measured "
            "not assumed: hiding B's own throttle output u from A raises grip 9x (1.09 -> 9.07) -- "
            "B's faithful report leaks ~88% of the hidden state through its own quantized bin. Any "
            "new mechanism proposed below must explain concretely why it would NOT leak the hidden "
            "state through its own output the same way -- 'don't reveal u' is not a valid answer, "
            "since A must observe SOME consequence of B's action for B to constrain A at all."
        ),
    },
    {
        "domain": "Deferred probe 1 -- parameter surface",
        "title": "Sweep (sigma_w, damage_penalty) instead of a single committed point",
        "description": (
            "The committed env (EnvConfig in env.py) fixes sigma_w=0.8352 (tuned so "
            "phi_static^H=0.50) and damage_penalty=2.0 (tuned for an interior optimum, a~0.4). "
            "Both were chosen once, pre-registered, never swept. Open question: does the "
            "engagement-dilemma leak (grip 9x when u hidden) hold across the (sigma_w, "
            "damage_penalty) surface, or is there a region where grip becomes material (>=10% of "
            "return) without sacrificing phi_eff persistence? Propose a concrete sub-region of this "
            "2D surface to test and a specific hypothesis for why it might behave differently from "
            "the single committed point already measured."
        ),
    },
    {
        "domain": "Deferred probe 2 -- B-revelation-fidelity frontier",
        "title": "Sweep K (throttle discretization) to map the engagement-dilemma frontier",
        "description": (
            "B's throttle currently uses K=5 discrete levels (env.py EnvConfig.K, "
            "throttle()/batch_step()). Coarser K (fewer levels) reveals less about z per "
            "intervention (lower resolution report) but may also reduce constraint fidelity CF "
            "(the violation-rate-reduction cross-check, tau_CF=50%). Finer K reveals more about z "
            "but may raise grip if A cannot exploit the extra resolution as easily. This is "
            "literally the fidelity-vs-opacity tradeoff the project's DESIGN.md Section 9 channel-"
            "capacity identity predicts (NA=H(V|O)=phi, no knee below full fidelity) -- but that "
            "identity was proven for the PASSIVE/static case; whether the same no-knee result holds "
            "for the CLOSED-LOOP grip metric (not just phi_eff) across a real K-sweep has not been "
            "measured. Propose a concrete K-sweep (e.g. K in {2,3,5,8,12}) and what pattern in grip "
            "vs K would indicate a genuine knee (a K where grip becomes material without CF dropping "
            "below tau_CF) versus confirm the no-knee prediction."
        ),
    },
    {
        "domain": "Deferred probe 3 -- information-seeking adversary",
        "title": "An A that actively tries to infer z, not just optimize reward",
        "description": (
            "The current agent (agent.py Policy, REINFORCE) optimizes throttled throughput only -- "
            "it has no auxiliary objective to actively infer z; any z-inference is an incidental "
            "byproduct of reward-seeking. An information-seeking A would add an explicit auxiliary "
            "objective (e.g. an auxiliary loss predicting u_{t+1} or z_t from its own observation "
            "history, trained jointly with or ahead of the policy) that rewards actions which reduce "
            "its own predictive uncertainty about the hidden state, independent of throughput reward "
            "-- closer to a genuinely adversarial/exploratory A than the current reward-only agent. "
            "Open question: does an A explicitly incentivized to reduce its OWN uncertainty about z "
            "(not just maximize return) collapse phi_eff^H further than the current reward-only "
            "agent does, or is phi_eff^H's persistence genuinely information-theoretic (from the "
            "exogenous w_t innovation) rather than an artifact of the current agent's objective? "
            "Propose a concrete, implementable auxiliary objective and how it plugs into the "
            "existing REINFORCE training loop (train_agent in agent.py) without requiring a new "
            "environment."
        ),
    },
]

DIRECTIVE0 = (
    "Propose concrete extensions to the existing ASDS kill-or-keep experiment (the four seed "
    "concepts above: the engagement-dilemma finding to engage with, and three deferred probes to "
    "pick from or combine). HARD RULES: "
    "(1) Any mechanism you propose MUST be implementable by modifying the EXISTING pure-numpy "
    "codebase (EnvConfig fields in env.py, FeatCfg toggles in agent.py, or the throttle()/"
    "batch_step() functions) -- do NOT propose a new environment, a different RL framework, or "
    "anything requiring dependencies beyond numpy. If your mechanism cannot be described as a "
    "specific code change to these existing files, it does not qualify. "
    "(2) You MUST explicitly state which of the three measured quantities (phi_effective, grip, "
    "CF) your proposal is expected to move, in which direction, and roughly why -- vague appeals to "
    "'more information' or 'better learning' without a specific causal mechanism do not qualify. "
    "(3) You MUST directly address the engagement dilemma: explain concretely why your proposed "
    "mechanism would NOT leak the hidden state through B's own output the same way the current "
    "faithful K=5 throttle does (measured: 88% grip leak). If you cannot explain this, say so "
    "explicitly as an open risk rather than ignoring it. "
    "(4) Any named theorem, algorithm, or control-theory/information-theory concept you invoke "
    "must be attributed or explicitly flagged as your own reasoning, not stated as established fact "
    "with no source. "
    "(5) Avoid vague biological/ecological or narrative metaphors (consciousness, sovereignty, "
    "will) beyond what the project's own DESIGN.md already uses precisely -- stay in the "
    "information-theoretic/control-theoretic register the existing measured results use."
)


async def main():
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
            engine = EvolutionEngine(session, directive=DIRECTIVE0, project=PROJECT)

            async def on_survivor(res, gen=gen):
                await persist_survivor(session, PROJECT, gen, res, log, score_fmt=".2f", show_prior_art=False, show_flaws=True)

            survivors = await engine.run_generation_cycle(gen, POPULATION, on_survivor=on_survivor)
            log.info(f"Gen {gen}: {len(survivors)} retained")
        break

    s = llm_engine.stats.summary
    log.info(f"LLM: {s['total_calls']} calls, {s.get('failed', 0)} failed, "
             f"{s['input_tokens'] + s['output_tokens']:,} tokens")


if __name__ == "__main__":
    asyncio.run(main())
