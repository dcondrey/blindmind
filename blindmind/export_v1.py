"""Export BlindMind concepts through the `crosstalk.blindmind.v1` contract.

The legacy `export` command emits a flat {"version": "1.0"} record with a single
fitness scalar. Crosstalk's reader needs the v1 shape: per-idea mutation type,
mechanism, predicted measurements, and kill criteria.

BlindMind's schema has no column for predicted measurements or kill criteria, so
this exporter omits those keys entirely rather than emitting an empty list that a
reader could mistake for "measured, found none". `Concept.description` is carried
as `mechanism`: it is a rename of the same field, not a reconstruction
(`MutationOutput.description` is specified as "the concept and its mechanics").
"""

import json
from datetime import UTC, datetime

from sqlmodel import select

from blindmind.db import get_async_session
from blindmind.models import Concept, Lineage

CONTRACT_VERSION = "crosstalk.blindmind.v1"

# Crosstalk's MutationOperator. REWRITE/TIGHTEN/AMPLIFY have no counterpart and
# make a concept unexportable rather than being folded into a neighbouring one.
OPERATOR_MAP = {
    "CROSSOVER": "Crossover",
    "POINT_MUTATION": "PointMutation",
    "INVERSION": "Inversion",
    "WILDCARD": "Wildcard",
}

SEED_OPERATOR = "Wildcard"


def build_export(concepts, lineages, project, directive):
    """Return (payload, report). Pure, so it is testable without a database."""
    by_id = {str(c.id): c for c in concepts}

    edges_outside_project = 0
    parents = {}
    operators = {}
    conflicting = set()
    for edge in lineages:
        child, parent = str(edge.child_id), str(edge.parent_id)
        if child not in by_id or parent not in by_id:
            edges_outside_project += 1
            continue
        parents.setdefault(child, []).append(parent)
        mutation = edge.mutation_type
        mutation = getattr(mutation, "value", mutation)
        if operators.setdefault(child, mutation) != mutation:
            conflicting.add(child)

    unexportable = {}

    def reject(concept_id, reason):
        unexportable.setdefault(concept_id, reason)

    for concept_id, concept in by_id.items():
        if not (concept.title or "").strip() or not (concept.domain or "").strip() or not (concept.description or "").strip():
            reject(concept_id, "missing a required v1 field (title, domain, or mechanism)")
        if concept_id in conflicting:
            reject(concept_id, "parents disagree on mutation type; v1 carries one operator per idea")
        mutation = operators.get(concept_id)
        if mutation is not None and mutation not in OPERATOR_MAP:
            reject(concept_id, f"mutation type {mutation} has no Crosstalk operator")

    # A Crosstalk lineage edge must run from a strictly earlier generation. This
    # binds the edge, not the concept: v1 imposes no generation ordering, so a
    # child whose recorded parent is not earlier is exported with its ancestry
    # truncated rather than dropped, and the loss is counted.
    unrepresentable_edges = 0
    truncated_children = set()
    for concept_id, concept in by_id.items():
        keep = [p for p in parents.get(concept_id, []) if by_id[p].generation < concept.generation]
        dropped = len(parents.get(concept_id, [])) - len(keep)
        if dropped:
            unrepresentable_edges += dropped
            truncated_children.add(concept_id)
            parents[concept_id] = keep

    # A dropped concept cannot remain a parent: Crosstalk rejects dangling refs.
    changed = True
    while changed:
        changed = False
        for concept_id in by_id:
            if concept_id in unexportable:
                continue
            if any(parent in unexportable for parent in parents.get(concept_id, [])):
                reject(concept_id, "an ancestor is unexportable, which would leave a dangling parent reference")
                changed = True

    exported = [c for cid, c in sorted(by_id.items()) if cid not in unexportable]
    exported.sort(key=lambda c: (c.generation, c.created_at, str(c.id)))

    ideas = []
    for concept in exported:
        concept_id = str(concept.id)
        parent_ids = sorted(parents.get(concept_id, []))
        idea = {
            "id": concept_id,
            "generation": concept.generation,
            "parent_ids": parent_ids,
            "mutation_type": OPERATOR_MAP.get(operators.get(concept_id, ""), SEED_OPERATOR),
            "domain": concept.domain,
            "title": concept.title,
            "mechanism": concept.description,
            "external_scores": {},
            "objectively_verified": False,
        }
        if concept.fitness_score is not None:
            idea["external_scores"]["blindmind_composite"] = concept.fitness_score
        if concept.tags:
            idea["tags"] = sorted({t.strip() for t in concept.tags.split(",") if t.strip()})
        ideas.append(idea)

    payload = {
        "schema": CONTRACT_VERSION,
        "project": project,
        "directive": directive or "",
        "exported_at": datetime.now(UTC).isoformat(),
        "ideas": ideas,
    }

    orphan_non_seeds = sum(
        1 for c in exported if c.generation > 0 and not parents.get(str(c.id))
    )
    report = {
        "project": project,
        "concepts_in": len(by_id),
        "concepts_exported": len(ideas),
        "unexportable": len(unexportable),
        "unexportable_reasons": _tally(unexportable.values()),
        "lineage_edges_in": len(lineages),
        "lineage_edges_exported": sum(len(i["parent_ids"]) for i in ideas),
        "lineage_edges_outside_project": edges_outside_project,
        "lineage_edges_dropped_unrepresentable": unrepresentable_edges,
        "children_with_truncated_ancestry": len(truncated_children),
        "non_seeds_without_recorded_parents": orphan_non_seeds,
        "concepts_with_a_fitness_scalar": sum(1 for i in ideas if i["external_scores"]),
        "concepts_with_predicted_measurements": 0,
        "concepts_with_kill_criteria": 0,
        "directive_recovered": bool(directive),
    }
    return payload, report


def _tally(reasons):
    counts = {}
    for reason in reasons:
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


async def export_v1_logic(file: str, project: str):
    async for session in get_async_session():
        concepts = (
            await session.execute(select(Concept).where(Concept.project == project))
        ).scalars().all()
        if not concepts:
            return None, {"project": project, "concepts_in": 0, "error": "no concepts in project"}

        concept_ids = {c.id for c in concepts}
        lineages = [
            link for link in (await session.execute(select(Lineage))).scalars().all()
            if link.child_id in concept_ids or link.parent_id in concept_ids
        ]

        from blindmind.models import EvolutionRun

        run = (
            await session.execute(
                select(EvolutionRun)
                .where(EvolutionRun.project == project)
                .order_by(EvolutionRun.updated_at.desc())
                .limit(1)
            )
        ).scalars().first()

        payload, report = build_export(concepts, lineages, project, run.latest_directive if run else None)
        with open(file, "w") as f:
            json.dump(payload, f, indent=2)
        return payload, report
