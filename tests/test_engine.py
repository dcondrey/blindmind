from unittest.mock import patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from blindmind.engine import EvolutionEngine
from blindmind.llm_schemas import CriticScore, MutationOutput
from blindmind.models import Concept


@pytest.fixture(name="session")
async def session_fixture():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

@pytest.mark.asyncio
async def test_run_generation_cycle(session: AsyncSession):
    c1 = Concept(domain="D1", title="T1", description="Desc1")
    c2 = Concept(domain="D2", title="T2", description="Desc2")
    session.add_all([c1, c2])
    await session.commit()

    engine = EvolutionEngine(session)

    mock_mutation = MutationOutput(title="New", domain="D", description="Desc", justification="J")
    mock_critique = CriticScore(conceptual_novelty=9, feasibility=9, utility=9, semantic_jump=8, rationale="Good", evolutionary_directive="Focus on tech")

    with patch("blindmind.engine.llm_engine.generate_mutation", return_value=mock_mutation), \
         patch("blindmind.engine.llm_engine.critique_mutation", return_value=mock_critique):

        survivors = await engine.run_generation_cycle(generation=0, population_size=2)
        assert len(survivors) == 2
        mutation, critique, parent_ids, m_type = survivors[0]
        assert mutation.title == "New"
        assert critique.conceptual_novelty == 9
        assert len(parent_ids) > 0

@pytest.mark.asyncio
async def test_run_generation_cycle_with_rejections(session: AsyncSession):
    c1 = Concept(domain="D1", title="T1", description="Desc1")
    session.add(c1)
    await session.commit()

    engine = EvolutionEngine(session)

    mock_mutation = MutationOutput(title="New", domain="D", description="Desc", justification="J")
    mock_critique_fail = CriticScore(conceptual_novelty=2, feasibility=2, utility=2, semantic_jump=1, rationale="Bad", evolutionary_directive="Try again")
    mock_critique_pass = CriticScore(conceptual_novelty=9, feasibility=9, utility=9, semantic_jump=9, rationale="Good", evolutionary_directive="Double down")

    with patch("blindmind.engine.llm_engine.generate_mutation", return_value=mock_mutation), \
         patch("blindmind.engine.llm_engine.critique_mutation", side_effect=[mock_critique_fail, mock_critique_pass]):

        survivors = await engine.run_generation_cycle(generation=0, population_size=1)
        assert len(survivors) == 1
        assert survivors[0][1].composite_score >= 7.0

@pytest.mark.asyncio
async def test_directive_passed_to_engine(session: AsyncSession):
    c1 = Concept(domain="D1", title="T1", description="Desc1")
    session.add(c1)
    await session.commit()

    directive = "Focus on biotech applications"
    engine = EvolutionEngine(session, directive=directive)
    assert engine.directive == directive

@pytest.mark.asyncio
async def test_configurable_mutation_rates(session: AsyncSession):
    engine = EvolutionEngine(session)
    assert engine.crossover_rate == 0.5
    assert engine.point_mutation_rate == 0.3

def test_rejection_memory():
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.rejected_titles = ["Quantum Blockchain Integration", "Neural Market Dynamics"]

    assert engine._is_too_similar_to_rejected("Quantum Blockchain Integration") is True
    assert engine._is_too_similar_to_rejected("quantum blockchain integration") is True
    assert engine._is_too_similar_to_rejected("Mycelial Computing") is False

def test_rejection_memory_word_overlap():
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.rejected_titles = ["Quantum Blockchain Neural Integration"]

    # High word overlap
    assert engine._is_too_similar_to_rejected("Quantum Neural Blockchain Integration") is True
    # Low word overlap
    assert engine._is_too_similar_to_rejected("Biological Swarm Computing") is False

def test_synthesize_directives_weighted():
    directives = [
        ("Focus on biology", 9.5),
        ("Explore physics", 7.0),
        ("Try economics", 8.2),
        ("Add art", 6.0),
    ]
    result = EvolutionEngine.synthesize_directives(directives)
    assert "Focus on biology" in result
    assert "Try economics" in result
    assert "Explore physics" in result
    assert "Add art" not in result

def test_synthesize_directives_empty():
    result = EvolutionEngine.synthesize_directives([])
    assert "novelty" in result.lower()

def test_fatal_flaws_penalize_but_do_not_zero_composite_score():
    """Listed flaws must knock a candidate below a normal (~7.0) retention threshold
    even when the other scores are inflated, but must not be an absolute zero-out:
    BlindMind is also run at a deliberately low threshold for raw, human-curated
    divergence (see headless_evolve.py), and a candidate the critic flagged should
    still be visible to that mode rather than erased."""
    flagged = CriticScore(
        conceptual_novelty=9, feasibility=9, utility=9, semantic_jump=9,
        fatal_flaws=["Violates conservation of energy"],
        rationale="High scores but broken", evolutionary_directive="Try again",
    )
    unflagged = CriticScore(
        conceptual_novelty=9, feasibility=9, utility=9, semantic_jump=9,
        rationale="High scores, no flaws", evolutionary_directive="Double down",
    )

    assert flagged.composite_score < unflagged.composite_score
    assert flagged.composite_score < 7.0
    assert flagged.composite_score > 0


@pytest.mark.asyncio
async def test_flagged_candidate_still_survives_a_permissive_threshold(session: AsyncSession):
    """Mirrors headless_evolve.py's critic_threshold=2.0 divergence mode: a flagged
    candidate should still come through when the human/downstream pipeline, not the
    critic, is meant to be the real filter."""
    c1 = Concept(domain="D1", title="T1", description="Desc1")
    session.add(c1)
    await session.commit()

    engine = EvolutionEngine(session)
    engine.adaptive_threshold = 2.0
    mock_mutation = MutationOutput(title="Flawed but interesting", domain="D", description="Desc", justification="J")
    mock_critique = CriticScore(
        conceptual_novelty=8, feasibility=7, utility=7, semantic_jump=6,
        fatal_flaws=["Unclear how step 2 works"],
        rationale="Promising but underspecified", evolutionary_directive="Sharpen the mechanism",
    )

    with patch("blindmind.engine.llm_engine.generate_mutation", return_value=mock_mutation), \
         patch("blindmind.engine.llm_engine.critique_mutation", return_value=mock_critique):
        survivors = await engine.run_generation_cycle(generation=0, population_size=1)

    assert len(survivors) == 1
    assert survivors[0][0].title == "Flawed but interesting"


def test_composite_score_with_prior_art():
    # Novel concept (low prior art overlap = high bonus)
    novel = CriticScore(conceptual_novelty=8, feasibility=7, utility=8, semantic_jump=6, prior_art_overlap=1, rationale="R", evolutionary_directive="D")
    # Derivative concept (high prior art overlap = low bonus)
    derivative = CriticScore(conceptual_novelty=8, feasibility=7, utility=8, semantic_jump=6, prior_art_overlap=9, rationale="R", evolutionary_directive="D")

    assert novel.composite_score > derivative.composite_score
