import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select

from blindmind.db import (
    delete_concept,
    get_diverse_parents,
    get_domain_distribution,
    get_latent_space_sample,
    get_random_concepts,
    get_stats,
    get_tournament_concepts,
    save_concept,
    search_concepts,
)
from blindmind.models import Concept, Lineage, MutationType


@pytest.fixture(name="session")
async def session_fixture():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.mark.asyncio
async def test_save_concept_with_lineage(session: AsyncSession):
    p1 = Concept(domain="Biology", title="Cell", description="Basic unit")
    session.add(p1)
    await session.commit()

    child = Concept(domain="Bio-Tech", title="Cyber-Cell", description="Enhanced cell", generation=1)
    await save_concept(session, child, parent_ids=[p1.id], mutation_type=MutationType.POINT_MUTATION)

    statement = select(Concept).where(Concept.title == "Cyber-Cell")
    result = (await session.execute(statement)).scalars().first()
    assert result.generation == 1

    lineage_stmt = select(Lineage).where(Lineage.child_id == child.id)
    lineage = (await session.execute(lineage_stmt)).scalars().first()
    assert lineage.parent_id == p1.id
    assert lineage.mutation_type == MutationType.POINT_MUTATION


@pytest.mark.asyncio
async def test_get_random_concepts(session: AsyncSession):
    for i in range(10):
        c = Concept(domain=f"Domain {i}", title=f"Title {i}", description=f"Desc {i}")
        session.add(c)
    await session.commit()

    randoms = await get_random_concepts(session, count=3)
    assert len(randoms) == 3
    assert len(set([r.id for r in randoms])) == 3


@pytest.mark.asyncio
async def test_tournament_deduplication(session: AsyncSession):
    for i in range(10):
        c = Concept(domain=f"D{i}", title=f"T{i}", description=f"Desc {i}", fitness_score=float(i))
        session.add(c)
    await session.commit()

    selected = await get_tournament_concepts(session, count=3)
    assert len(selected) == 3
    ids = [c.id for c in selected]
    assert len(set(ids)) == 3


@pytest.mark.asyncio
async def test_search_by_keyword(session: AsyncSession):
    c1 = Concept(domain="Biology", title="Quantum Biology", description="Quantum effects in cells")
    c2 = Concept(domain="Physics", title="Newtonian Mechanics", description="Classical physics laws")
    c3 = Concept(domain="Tech", title="Quantum Computing", description="Using qubits for computation")
    session.add_all([c1, c2, c3])
    await session.commit()

    results = await search_concepts(session, query="quantum")
    assert len(results) == 2
    titles = {r.title for r in results}
    assert "Quantum Biology" in titles
    assert "Quantum Computing" in titles


@pytest.mark.asyncio
async def test_search_by_domain(session: AsyncSession):
    c1 = Concept(domain="Biology", title="Cell", description="Basic unit")
    c2 = Concept(domain="Physics", title="Gravity", description="Attractive force")
    session.add_all([c1, c2])
    await session.commit()

    results = await search_concepts(session, domain="biology")
    assert len(results) == 1
    assert results[0].title == "Cell"


@pytest.mark.asyncio
async def test_search_by_fitness_range(session: AsyncSession):
    c1 = Concept(domain="D1", title="Low", description="d", fitness_score=3.0)
    c2 = Concept(domain="D2", title="Mid", description="d", fitness_score=6.0)
    c3 = Concept(domain="D3", title="High", description="d", fitness_score=9.0)
    session.add_all([c1, c2, c3])
    await session.commit()

    results = await search_concepts(session, min_fitness=5.0)
    assert len(results) == 2
    titles = {r.title for r in results}
    assert "Mid" in titles
    assert "High" in titles


@pytest.mark.asyncio
async def test_search_by_tags(session: AsyncSession):
    c1 = Concept(domain="D1", title="T1", description="d", tags="biology,quantum")
    c2 = Concept(domain="D2", title="T2", description="d", tags="physics,gravity")
    session.add_all([c1, c2])
    await session.commit()

    results = await search_concepts(session, tags="quantum")
    assert len(results) == 1
    assert results[0].title == "T1"


@pytest.mark.asyncio
async def test_get_stats(session: AsyncSession):
    c1 = Concept(domain="Biology", title="Cell", description="d", generation=0)
    c2 = Concept(domain="Physics", title="Gravity", description="d", generation=0)
    c3 = Concept(domain="Bio-Tech", title="Cyber-Cell", description="d", generation=1, fitness_score=8.5)
    session.add_all([c1, c2, c3])
    await session.commit()

    stats = await get_stats(session)
    assert stats["total"] == 3
    assert stats["seeds"] == 2
    assert stats["evolved"] == 1
    assert stats["domains"] == 3
    assert stats["generations"] == 1
    assert stats["avg_fitness"] == 8.5
    assert stats["max_fitness"] == 8.5


@pytest.mark.asyncio
async def test_delete_concept(session: AsyncSession):
    p1 = Concept(domain="D1", title="Parent", description="d")
    session.add(p1)
    await session.commit()

    child = Concept(domain="D2", title="Child", description="d", generation=1)
    await save_concept(session, child, parent_ids=[p1.id], mutation_type=MutationType.CROSSOVER)

    result = await delete_concept(session, child.id)
    assert result is True

    # Verify concept deleted
    check = (await session.execute(select(Concept).where(Concept.id == child.id))).scalars().first()
    assert check is None

    # Verify lineage deleted
    lin = (await session.execute(select(Lineage).where(Lineage.child_id == child.id))).scalars().first()
    assert lin is None

    # Parent should still exist
    parent_check = (await session.execute(select(Concept).where(Concept.id == p1.id))).scalars().first()
    assert parent_check is not None


@pytest.mark.asyncio
async def test_delete_nonexistent_concept(session: AsyncSession):
    from uuid import uuid4

    result = await delete_concept(session, uuid4())
    assert result is False


@pytest.mark.asyncio
async def test_stats_empty_db(session: AsyncSession):
    stats = await get_stats(session)
    assert stats["total"] == 0
    assert stats["avg_fitness"] == 0


@pytest.mark.asyncio
async def test_diverse_parents_prefer_different_domains(session: AsyncSession):
    for i in range(10):
        c = Concept(domain=f"Domain{i % 3}", title=f"T{i}", description=f"D{i}", fitness_score=float(i))
        session.add(c)
    await session.commit()

    parents = await get_diverse_parents(session, count=2)
    assert len(parents) == 2
    # Should prefer different domains
    assert parents[0].domain != parents[1].domain


@pytest.mark.asyncio
async def test_get_domain_distribution(session: AsyncSession):
    session.add_all(
        [
            Concept(domain="Biology", title="T1", description="d"),
            Concept(domain="Biology", title="T2", description="d"),
            Concept(domain="Physics", title="T3", description="d"),
        ]
    )
    await session.commit()

    dist = await get_domain_distribution(session)
    assert dist["Biology"] == 2
    assert dist["Physics"] == 1


@pytest.mark.asyncio
async def test_latent_space_sample(session: AsyncSession):
    for i in range(20):
        c = Concept(domain=f"D{i}", title=f"T{i}", description=f"Desc{i}", fitness_score=float(i))
        session.add(c)
    await session.commit()

    sample = await get_latent_space_sample(session, limit=10)
    assert len(sample) <= 10
    assert len(set(c.id for c in sample)) == len(sample)
