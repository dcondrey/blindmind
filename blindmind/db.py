import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.expression import func
from sqlmodel import SQLModel, select

from blindmind.config import settings
from blindmind.logging import logger
from blindmind.models import Concept, EvolutionRun, Lineage

# Ensure data directory exists
db_path = settings.database_url.replace("sqlite+aiosqlite:///", "")
if db_path and "/" in db_path:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

engine = create_async_engine(settings.database_url)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    logger.info("Initializing database...")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await _migrate_add_project_column()

async def _migrate_add_project_column():
    async with engine.begin() as conn:
        try:
            await conn.execute(text("SELECT project FROM concept LIMIT 1"))
        except Exception:
            await conn.execute(text("ALTER TABLE concept ADD COLUMN project VARCHAR DEFAULT 'default'"))
            logger.info("Migrated: added 'project' column to concept table")
        try:
            await conn.execute(text("SELECT project FROM evolutionrun LIMIT 1"))
        except Exception:
            await conn.execute(text("ALTER TABLE evolutionrun ADD COLUMN project VARCHAR DEFAULT 'default'"))
            logger.info("Migrated: added 'project' column to evolutionrun table")

async def get_async_session() -> AsyncSession:
    async with async_session() as session:
        yield session

async def save_concept(session: AsyncSession, concept: Concept, parent_ids: list = None, mutation_type=None):
    session.add(concept)
    await session.commit()
    await session.refresh(concept)

    if parent_ids:
        for p_id in parent_ids:
            lineage = Lineage(child_id=concept.id, parent_id=p_id, mutation_type=mutation_type)
            session.add(lineage)
        await session.commit()

    return concept

async def get_random_concepts(session: AsyncSession, count: int, generation: int = None, project: str = None):
    statement = select(Concept)
    if project:
        statement = statement.where(Concept.project == project)
    if generation is not None:
        statement = statement.where(Concept.generation == generation)

    statement = statement.order_by(func.random()).limit(count)
    result = await session.execute(statement)
    return result.scalars().all()

async def get_tournament_concepts(session: AsyncSession, count: int, tournament_size: int = 5, project: str = None):
    selected = []
    seen_ids = set()
    for _ in range(count):
        sample_stmt = select(Concept)
        if project:
            sample_stmt = sample_stmt.where(Concept.project == project)
        sample_stmt = sample_stmt.order_by(func.random()).limit(tournament_size)
        pool = (await session.execute(sample_stmt)).scalars().all()

        if not pool:
            break

        pool.sort(key=lambda c: c.fitness_score or 0.0, reverse=True)
        best = next((c for c in pool if c.id not in seen_ids), pool[0])
        selected.append(best)
        seen_ids.add(best.id)

    return selected

async def create_run(
    session: AsyncSession,
    config_json: str = None,
    total_generations: int = 1,
    population_size: int = 5,
    project: str = "default",
) -> EvolutionRun:
    run = EvolutionRun(
        config_snapshot=config_json,
        total_generations=total_generations,
        population_size=population_size,
        project=project,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run

async def search_concepts(
    session: AsyncSession,
    query: str = None,
    domain: str = None,
    min_fitness: float = None,
    max_fitness: float = None,
    generation: int = None,
    tags: str = None,
    limit: int = 20,
    project: str = None,
):
    statement = select(Concept)

    if project:
        statement = statement.where(Concept.project == project)
    if query:
        pattern = f"%{query}%"
        statement = statement.where(
            (Concept.title.ilike(pattern)) | (Concept.description.ilike(pattern))
        )
    if domain:
        statement = statement.where(Concept.domain.ilike(f"%{domain}%"))
    if min_fitness is not None:
        statement = statement.where(Concept.fitness_score >= min_fitness)
    if max_fitness is not None:
        statement = statement.where(Concept.fitness_score <= max_fitness)
    if generation is not None:
        statement = statement.where(Concept.generation == generation)
    if tags:
        for tag in tags.split(","):
            statement = statement.where(Concept.tags.ilike(f"%{tag.strip()}%"))

    statement = statement.order_by(Concept.created_at.desc()).limit(limit)
    result = await session.execute(statement)
    return result.scalars().all()

async def get_stats(session: AsyncSession, project: str = None) -> dict:
    base = select(func.count(Concept.id))
    if project:
        base = base.where(Concept.project == project)
    total = (await session.execute(base)).scalar() or 0
    if total == 0:
        return {
            "total": 0,
            "generations": 0,
            "avg_fitness": 0,
            "max_fitness": 0,
            "min_fitness": 0,
            "domains": 0,
            "seeds": 0,
            "evolved": 0,
        }

    def _where(stmt):
        if project:
            return stmt.where(Concept.project == project)
        return stmt

    max_gen = (await session.execute(_where(select(func.max(Concept.generation))))).scalar() or 0
    avg_fitness = (
        await session.execute(
            _where(select(func.avg(Concept.fitness_score)).where(Concept.fitness_score.isnot(None)))
        )
    ).scalar() or 0
    max_fitness = (await session.execute(_where(select(func.max(Concept.fitness_score))))).scalar() or 0
    min_fitness = (
        await session.execute(
            _where(select(func.min(Concept.fitness_score)).where(Concept.fitness_score.isnot(None)))
        )
    ).scalar() or 0
    domains = (await session.execute(_where(select(func.count(func.distinct(Concept.domain)))))).scalar() or 0
    seeds = (await session.execute(_where(select(func.count(Concept.id)).where(Concept.generation == 0)))).scalar() or 0
    evolved = total - seeds

    gen_rows = (await session.execute(
        _where(select(Concept.generation, func.count(Concept.id)).group_by(Concept.generation))
    )).all()
    gen_dist = dict.fromkeys(range(max_gen + 1), 0)
    gen_dist.update({gen: count for gen, count in gen_rows})

    return {
        "total": total, "generations": max_gen, "avg_fitness": round(avg_fitness, 2),
        "max_fitness": round(max_fitness, 2), "min_fitness": round(min_fitness, 2),
        "domains": domains, "seeds": seeds, "evolved": evolved, "gen_distribution": gen_dist,
    }

async def get_domain_distribution(session: AsyncSession, project: str = None) -> dict:
    stmt = select(Concept)
    if project:
        stmt = stmt.where(Concept.project == project)
    concepts = (await session.execute(stmt)).scalars().all()
    dist = {}
    for c in concepts:
        dist[c.domain] = dist.get(c.domain, 0) + 1
    return dist

async def get_diverse_parents(
    session: AsyncSession, count: int = 2, tournament_size: int = 5, project: str = None
) -> list:
    selected = []
    used_domains = set()

    for _ in range(count):
        sample_stmt = select(Concept)
        if project:
            sample_stmt = sample_stmt.where(Concept.project == project)
        sample_stmt = sample_stmt.order_by(func.random()).limit(tournament_size * 2)
        pool = (await session.execute(sample_stmt)).scalars().all()
        if not pool:
            break

        novel_pool = [c for c in pool if c.domain not in used_domains]
        candidates = novel_pool if novel_pool else pool

        candidates.sort(key=lambda c: c.fitness_score or 0.0, reverse=True)
        best = candidates[0]
        selected.append(best)
        used_domains.add(best.domain)

    return selected

async def get_latent_space_sample(session: AsyncSession, limit: int = 15, project: str = None) -> list:
    def _proj(stmt):
        if project:
            return stmt.where(Concept.project == project)
        return stmt

    top_fitness = (await session.execute(
        _proj(select(Concept).where(Concept.fitness_score.isnot(None))).order_by(Concept.fitness_score.desc()).limit(5)
    )).scalars().all()

    recent = (await session.execute(
        _proj(select(Concept)).order_by(Concept.created_at.desc()).limit(5)
    )).scalars().all()

    random_sample = (await session.execute(
        _proj(select(Concept)).order_by(func.random()).limit(5)
    )).scalars().all()

    seen = set()
    result = []
    for c in list(top_fitness) + list(recent) + list(random_sample):
        if c.id not in seen:
            seen.add(c.id)
            result.append(c)
        if len(result) >= limit:
            break
    return result

async def get_projects(session: AsyncSession) -> list:
    result = await session.execute(select(func.distinct(Concept.project)))
    return [r[0] for r in result.all()]

async def delete_concept(session: AsyncSession, concept_id) -> bool:
    concept = (await session.execute(select(Concept).where(Concept.id == concept_id))).scalars().first()
    if not concept:
        return False

    lineages = (await session.execute(
        select(Lineage).where((Lineage.child_id == concept_id) | (Lineage.parent_id == concept_id))
    )).scalars().all()
    for link in lineages:
        await session.delete(link)

    await session.delete(concept)
    await session.commit()
    return True
