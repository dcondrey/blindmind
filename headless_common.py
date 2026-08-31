"""Shared helper for the headless_evolve_*.py scripts.

Passing `persist_survivor` as EvolutionEngine.run_generation_cycle's `on_survivor`
callback saves each survivor immediately as it's found, instead of waiting for the
whole (possibly multi-batch, long-running) generation cycle to return and then
persisting the accumulated list. Without it, a run killed mid-cycle (SIGKILL,
KeyboardInterrupt, an unhandled exception, or a tool/process timeout) discards every
survivor found so far, because none of them were saved yet -- this has already
happened in production (see headless_evolve_erdos_gen1_only.py's header comment).
"""

from blindmind.db import save_concept
from blindmind.models import Concept


async def persist_survivor(session, project, gen, res, log, score_fmt=".1f", show_prior_art=True, show_flaws=False):
    """Save one survivor immediately and log it.

    Returns (evolutionary_directive, composite_score), the pair EvolutionEngine.
    synthesize_directives expects, so callers can keep accumulating it themselves.
    """
    mutation, critique, parent_ids, m_type = res
    mt = getattr(m_type, "value", str(m_type))
    tags = f"gen{gen},{mt},priorart{critique.prior_art_overlap},nov{critique.conceptual_novelty}"
    concept = Concept(
        project=project,
        domain=mutation.domain,
        title=mutation.title,
        description=mutation.description,
        generation=gen,
        fitness_score=critique.composite_score,
        tags=tags,
    )
    await save_concept(session, concept, parent_ids=parent_ids, mutation_type=m_type)

    msg = f"  kept [{mutation.domain}] {mutation.title} (score {critique.composite_score:{score_fmt}}"
    if show_prior_art:
        msg += f", priorArt {critique.prior_art_overlap}/10"
    if show_flaws:
        msg += f", flaws={len(critique.fatal_flaws)}"
    msg += ")"
    log.info(msg)

    return critique.evolutionary_directive, critique.composite_score
