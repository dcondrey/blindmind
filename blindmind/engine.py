import asyncio
import random
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from blindmind.config import settings
from blindmind.db import (
    get_diverse_parents,
    get_domain_distribution,
    get_latent_space_sample,
    get_tournament_concepts,
)
from blindmind.llm import llm_engine
from blindmind.llm_schemas import CriticScore, MutationOutput
from blindmind.logging import logger
from blindmind.models import Concept, MutationType
from blindmind.prompts import CRITIC_PROMPT, CROSSOVER_PROMPT, INVERSION_PROMPT, POINT_MUTATION_PROMPT, WILDCARD_PROMPT

GENERIC_TITLE_PLACEHOLDERS = {"untitled", "concept", "new concept", "n/a", "todo", "idea", "placeholder"}

ALL_DOMAINS = [
    "Biology",
    "Physics",
    "Chemistry",
    "Mathematics",
    "Computer Science",
    "Economics",
    "Philosophy",
    "Psychology",
    "Sociology",
    "Neuroscience",
    "Engineering",
    "Medicine",
    "Ecology",
    "Linguistics",
    "Art",
    "Music",
    "Architecture",
    "Law",
    "Education",
    "Anthropology",
]


class EvolutionEngine:
    def __init__(self, session: AsyncSession, directive: str | None = None, project: str = "default"):
        self.session = session
        self.directive = directive
        self.project = project
        self.crossover_rate = settings.crossover_rate
        self.point_mutation_rate = settings.point_mutation_rate
        self.inversion_rate = settings.inversion_rate
        self.rejected_titles: list[str] = []
        self.adaptive_threshold = settings.critic_threshold
        # Last exception raised while creating a candidate (generate_mutation or
        # critique_mutation failing outright, e.g. no configured provider, all
        # providers exhausted). run_generation_cycle otherwise just returns an empty
        # survivors list, which reads identically to "everyone scored below
        # threshold" -- this lets the caller (cli.py) tell those two very different
        # situations apart and surface the real cause instead of a misleading
        # "try lowering the threshold" suggestion.
        self.last_error: str | None = None
        self._latent_context = None
        self._existing_titles = None
        self._domain_dist = None
        # AsyncSession is not safe for concurrent use: run_generation_cycle fans
        # _create_candidate out via asyncio.gather, and several of those coroutines
        # call this shared session (get_diverse_parents/get_tournament_concepts)
        # concurrently. That's undefined behavior with SQLAlchemy's async session
        # (can corrupt session state or, with aiosqlite's thread-bridge, stall an
        # awaited future indefinitely). Serialize only the DB calls; LLM calls stay
        # concurrent (they're independently gated by llm_engine's own semaphore).
        self._session_lock = asyncio.Lock()

    async def _load_context(self):
        if self._latent_context is None:
            self._latent_context = await get_latent_space_sample(self.session, project=self.project)
            stmt = select(Concept).where(Concept.project == self.project)
            all_concepts = (await self.session.execute(stmt)).scalars().all()
            self._existing_titles = [f"[{c.domain}] {c.title}" for c in all_concepts]
            self._domain_dist = await get_domain_distribution(self.session, project=self.project)

    def _get_underrepresented_domains(self) -> list[str]:
        if not self._domain_dist:
            return ALL_DOMAINS[:5]
        existing = set(self._domain_dist.keys())
        missing = [d for d in ALL_DOMAINS if d not in existing]
        if missing:
            return missing[:5]
        avg = sum(self._domain_dist.values()) / len(self._domain_dist)
        return [d for d, c in self._domain_dist.items() if c < avg][:5]

    async def run_generation_cycle(
        self,
        generation: int,
        population_size: int,
        on_survivor: (
            Callable[[tuple[MutationOutput, CriticScore, list[UUID], MutationType]], Awaitable[None]] | None
        ) = None,
    ) -> list[tuple[MutationOutput, CriticScore, list[UUID], MutationType]]:
        logger.info(f"Starting generation {generation} cycle. Target population: {population_size}")
        if self.directive:
            logger.info(f"Active Directive: {self.directive}")

        await self._load_context()

        survivors = []
        max_attempts = population_size * 5
        attempts = 0
        consecutive_failures = 0

        while len(survivors) < population_size and attempts < max_attempts:
            needed = population_size - len(survivors)
            batch_size = min(needed * 2, 10)

            tasks = [self._create_candidate(generation) for _ in range(batch_size)]
            results = await asyncio.gather(*tasks)

            batch_successes = 0
            for res in results:
                if res:
                    mutation, critique, parent_ids, m_type = res
                    if critique.composite_score >= self.adaptive_threshold:
                        survivors.append(res)
                        batch_successes += 1
                        logger.info(
                            f"Survivor: '{mutation.title}' (score={critique.composite_score:.2f}, type={m_type})"
                        )
                        if on_survivor is not None:
                            # Persist immediately: an outer per-generation timeout
                            # cancels this coroutine mid-loop, and a survivor only
                            # held in the local `survivors` list would be lost with
                            # it. Saving here makes already-found work durable
                            # regardless of whether this generation finishes.
                            await on_survivor(res)
                        if len(survivors) >= population_size:
                            break
                    else:
                        if critique.fatal_flaws:
                            flaws = "; ".join(critique.fatal_flaws)
                            logger.debug(
                                f"Rejected '{mutation.title}' (score={critique.composite_score:.2f}, flaws: {flaws})"
                            )
                        self.rejected_titles.append(mutation.title)
                        consecutive_failures += 1

            if batch_successes > 0:
                consecutive_failures = 0

            # Adaptive threshold: lower if struggling, raise if too easy
            if consecutive_failures >= batch_size * 2 and self.adaptive_threshold > 4.0:
                old = self.adaptive_threshold
                self.adaptive_threshold = max(4.0, self.adaptive_threshold - 0.5)
                logger.info(f"Adaptive pressure: threshold lowered {old:.1f} -> {self.adaptive_threshold:.1f}")
                consecutive_failures = 0
            elif batch_successes == batch_size and self.adaptive_threshold < 9.0:
                old = self.adaptive_threshold
                self.adaptive_threshold = min(9.0, self.adaptive_threshold + 0.3)
                logger.info(f"Adaptive pressure: threshold raised {old:.1f} -> {self.adaptive_threshold:.1f}")

            attempts += batch_size
            logger.info(
                f"Progress: {len(survivors)}/{population_size} survivors "
                f"(threshold={self.adaptive_threshold:.1f}, attempts={attempts})"
            )

        return survivors

    async def _create_candidate(
        self, generation: int
    ) -> tuple[MutationOutput, CriticScore, list[UUID], MutationType] | None:
        try:
            choice = random.random()

            # 10% chance of wildcard when we have enough concepts
            if choice < 0.1 and self._latent_context and len(self._latent_context) >= 4:
                return await self._create_wildcard()
            elif choice < 0.1 + self.crossover_rate:
                # Use diversity-aware parent selection for crossover
                async with self._session_lock:
                    parents = await get_diverse_parents(self.session, count=2, project=self.project)
                if len(parents) < 2:
                    return await self._create_point_mutation()
                parent_ids = [p.id for p in parents]
                prompt = CROSSOVER_PROMPT.render(
                    concepts=parents,
                    directive=self.directive,
                    latent_space_context=self._latent_context,
                )
                m_type = MutationType.CROSSOVER
            elif choice < 0.1 + self.crossover_rate + self.point_mutation_rate:
                return await self._create_point_mutation()
            elif choice < 0.1 + self.crossover_rate + self.point_mutation_rate + self.inversion_rate:
                async with self._session_lock:
                    parents = await get_tournament_concepts(self.session, count=1, project=self.project)
                if not parents:
                    return None
                parent_ids = [p.id for p in parents]
                prompt = INVERSION_PROMPT.render(
                    concept=parents[0],
                    directive=self.directive,
                    latent_space_context=self._latent_context,
                )
                m_type = MutationType.INVERSION
            else:
                # Any probability mass left over once wildcard/crossover/point-mutation/
                # inversion rates are all accounted for falls back to point mutation.
                return await self._create_point_mutation()

            mutation = await llm_engine.generate_mutation(prompt)

            reject_reason = self._prefilter_reject(mutation)
            if reject_reason:
                logger.debug(f"Pre-filtered '{mutation.title}' before critique ({reject_reason})")
                self.rejected_titles.append(mutation.title)
                return None

            critique = await llm_engine.critique_mutation(
                CRITIC_PROMPT.render(mutation=mutation, existing_titles=self._existing_titles)
            )
            return mutation, critique, parent_ids, m_type

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error creating candidate: {e}")
            self.last_error = str(e)
            return None

    async def _create_point_mutation(self) -> tuple[MutationOutput, CriticScore, list[UUID], MutationType] | None:
        async with self._session_lock:
            parents = await get_tournament_concepts(self.session, count=1, project=self.project)
        if not parents:
            return None
        parent_ids = [parents[0].id]
        prompt = POINT_MUTATION_PROMPT.render(
            concept=parents[0],
            directive=self.directive,
            latent_space_context=self._latent_context,
        )
        mutation = await llm_engine.generate_mutation(prompt)

        reject_reason = self._prefilter_reject(mutation)
        if reject_reason:
            logger.debug(f"Pre-filtered '{mutation.title}' before critique ({reject_reason})")
            self.rejected_titles.append(mutation.title)
            return None

        critique = await llm_engine.critique_mutation(
            CRITIC_PROMPT.render(mutation=mutation, existing_titles=self._existing_titles)
        )
        return mutation, critique, parent_ids, MutationType.POINT_MUTATION

    async def _create_wildcard(self) -> tuple[MutationOutput, CriticScore, list[UUID], MutationType] | None:
        prompt = WILDCARD_PROMPT.render(
            directive=self.directive,
            latent_space_context=self._latent_context,
            underrepresented_domains=self._get_underrepresented_domains(),
        )
        mutation = await llm_engine.generate_mutation(prompt)

        reject_reason = self._prefilter_reject(mutation)
        if reject_reason:
            logger.debug(f"Pre-filtered '{mutation.title}' before critique ({reject_reason})")
            self.rejected_titles.append(mutation.title)
            return None

        critique = await llm_engine.critique_mutation(
            CRITIC_PROMPT.render(mutation=mutation, existing_titles=self._existing_titles)
        )
        return mutation, critique, [], MutationType.WILDCARD

    def _is_degenerate(self, mutation: MutationOutput) -> bool:
        """Catches structurally empty/placeholder candidates -- not just substantively
        weak ones -- so they never cost a critique round-trip at all."""
        title = (mutation.title or "").strip()
        desc = (mutation.description or "").strip()
        if not title or not desc:
            return True
        if title.lower() in GENERIC_TITLE_PLACEHOLDERS:
            return True
        return False

    def _find_near_duplicate(self, title: str) -> str | None:
        """Fuzzy-matches a candidate title against both prior critic rejections and
        concepts already saved in this project (self._existing_titles), using
        difflib.SequenceMatcher so paraphrased near-repeats -- not just literal
        substrings -- are caught before the second LLM round-trip (critique_mutation).
        Returns the matched title, or None."""
        title_norm = (title or "").strip().lower()
        if not title_norm:
            return None
        pool = list(self.rejected_titles[-50:])
        if self._existing_titles:
            # Entries look like "[Domain] Title"; compare against the title part only.
            pool += [t.split("] ", 1)[-1] for t in self._existing_titles]
        for other in pool:
            other_norm = (other or "").strip().lower()
            if not other_norm:
                continue
            if title_norm == other_norm:
                return other
            if SequenceMatcher(None, title_norm, other_norm).ratio() > 0.85:
                return other
        return None

    def _prefilter_reject(self, mutation: MutationOutput) -> str | None:
        """Local, dependency-free pre-critique filter. Rejects degenerate or
        near-duplicate candidates without spending the second LLM round-trip
        (critique_mutation) on them. Returns a short rejection reason, or None if
        the candidate should proceed to critique."""
        if self._is_degenerate(mutation):
            return "degenerate (empty/near-empty description or placeholder title)"
        dup = self._find_near_duplicate(mutation.title)
        if dup:
            return f"near-duplicate of existing/rejected title '{dup}'"
        if self._is_too_similar_to_rejected(mutation.title):
            return "high word-overlap with a recently rejected title"
        return None

    def _is_too_similar_to_rejected(self, title: str) -> bool:
        title_lower = title.lower()
        for rejected in self.rejected_titles[-50:]:
            if title_lower == rejected.lower():
                return True
            # Check word overlap
            title_words = set(title_lower.split())
            rejected_words = set(rejected.lower().split())
            if len(title_words) > 1 and len(rejected_words) > 1:
                overlap = len(title_words & rejected_words) / max(len(title_words | rejected_words), 1)
                if overlap > 0.7:
                    return True
        return False

    @staticmethod
    def synthesize_directives(directives: list[tuple[str, float]]) -> str:
        """Weight directives by their critique scores, pick the top ones."""
        if not directives:
            return "Focus on maximizing combinatorial novelty and logical consistency."
        sorted_dirs = sorted(directives, key=lambda x: x[1], reverse=True)
        top = sorted_dirs[:3]
        return " ".join(d for d, _ in top)
