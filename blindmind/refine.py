import asyncio
import os
import random
from typing import List, Tuple, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from blindmind.llm import llm_engine
from blindmind.essay_prompts import (
    STYLE_GUIDE, PARSE_PROMPT, REWRITE_PROMPT, TIGHTEN_PROMPT,
    AMPLIFY_PROMPT, CROSSOVER_PROMPT, INVERSION_PROMPT, CRITIQUE_PROMPT,
    COHERENCE_CHECK_PROMPT,
)
from blindmind.essay_schemas import SectionParse, SectionRewrite, EssayCriticScore, CoherenceCheck
from blindmind.models import Concept, MutationType
from blindmind.db import save_concept, get_async_session
from blindmind.config import settings
from blindmind.logging import logger


class EssayRefiner:
    def __init__(self, session: AsyncSession, project: str, style_guide: str = None):
        self.session = session
        self.project = project
        self.style_guide = style_guide or STYLE_GUIDE
        self.thesis = ""
        self.sections: List[dict] = []

    async def parse_essay(self, essay_text: str) -> List[dict]:
        self.thesis = essay_text[:500]

        prompt = PARSE_PROMPT.render(essay_text=essay_text)
        parsed = await llm_engine._completion(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format=SectionParse,
        )

        self.sections = []
        for i, section in enumerate(parsed.sections):
            self.sections.append({
                "index": i,
                "title": section.title,
                "function": section.function,
                "content": section.content,
                "concept_id": None,
            })

        return self.sections

    async def seed_sections(self) -> List[Concept]:
        concepts = []
        for sec in self.sections:
            concept = Concept(
                project=self.project,
                domain=f"Section {sec['index'] + 1}",
                title=sec["title"],
                description=sec["content"],
                generation=0,
                tags=f"function:{sec['function']},section:{sec['index']}",
            )
            await save_concept(self.session, concept)
            sec["concept_id"] = concept.id
            concepts.append(concept)
        return concepts

    def _get_context(self, section_index: int, selected_content: dict = None) -> dict:
        contents = selected_content or {i: s["content"] for i, s in enumerate(self.sections)}
        return {
            "thesis": self.thesis,
            "style_guide": self.style_guide,
            "section_function": self.sections[section_index]["function"],
            "section_content": contents[section_index],
            "preceding_section": contents.get(section_index - 1, ""),
            "following_section": contents.get(section_index + 1, ""),
        }

    async def generate_variant(self, section_index: int, mutation_type: str, temperature: float = 1.0, selected_content: dict = None, version_b: str = None) -> Tuple[str, str, EssayCriticScore]:
        ctx = self._get_context(section_index, selected_content)

        if mutation_type == "REWRITE":
            prompt = REWRITE_PROMPT.render(**ctx)
        elif mutation_type == "TIGHTEN":
            prompt = TIGHTEN_PROMPT.render(**ctx)
        elif mutation_type == "AMPLIFY":
            prompt = AMPLIFY_PROMPT.render(**ctx)
        elif mutation_type == "INVERSION":
            prompt = INVERSION_PROMPT.render(**ctx)
        elif mutation_type == "CROSSOVER" and version_b:
            prompt = CROSSOVER_PROMPT.render(
                style_guide=self.style_guide,
                thesis=self.thesis,
                version_a=ctx["section_content"],
                version_b=version_b,
                section_function=ctx["section_function"],
                preceding_section=ctx["preceding_section"],
                following_section=ctx["following_section"],
            )
        else:
            prompt = REWRITE_PROMPT.render(**ctx)

        rewrite = await llm_engine._completion(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format=SectionRewrite,
        )

        critique_prompt = CRITIQUE_PROMPT.render(
            style_guide=self.style_guide,
            thesis=self.thesis,
            section_function=ctx["section_function"],
            section_content=rewrite.content,
            preceding_section=ctx["preceding_section"],
            following_section=ctx["following_section"],
        )
        critique = await llm_engine._completion(
            [{"role": "user", "content": critique_prompt}],
            temperature=0.1,
            response_format=EssayCriticScore,
        )

        return rewrite.content, rewrite.approach, critique

    async def evolve_section(self, section_index: int, num_variants: int = 3, generation: int = 1, temperature: float = 1.0, selected_content: dict = None) -> List[Tuple[str, str, EssayCriticScore, str]]:
        mutation_types = ["REWRITE", "TIGHTEN", "AMPLIFY", "INVERSION"]
        variants = []

        tasks = []
        chosen_types = []
        for _ in range(num_variants):
            mt = random.choice(mutation_types)
            chosen_types.append(mt)
            tasks.append(self.generate_variant(section_index, mt, temperature=temperature, selected_content=selected_content))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for mt, result in zip(chosen_types, results):
            if isinstance(result, Exception):
                logger.error(f"Variant generation failed: {result}")
                continue
            content, approach, critique = result
            variants.append((content, approach, critique, mt))

        return variants

    async def save_variant(self, section_index: int, content: str, generation: int, fitness: float, mutation_type: str, parent_id: UUID = None) -> Concept:
        sec = self.sections[section_index]
        mt_map = {
            "REWRITE": MutationType.REWRITE,
            "TIGHTEN": MutationType.TIGHTEN,
            "AMPLIFY": MutationType.AMPLIFY,
            "INVERSION": MutationType.INVERSION,
            "CROSSOVER": MutationType.CROSSOVER,
        }
        concept = Concept(
            project=self.project,
            domain=f"Section {section_index + 1}",
            title=sec["title"],
            description=content,
            generation=generation,
            fitness_score=fitness,
            tags=f"function:{sec['function']},section:{section_index}",
        )
        parent_ids = [parent_id] if parent_id else ([sec["concept_id"]] if sec["concept_id"] else [])
        await save_concept(
            self.session, concept,
            parent_ids=parent_ids,
            mutation_type=mt_map.get(mutation_type, MutationType.REWRITE),
        )
        return concept

    async def check_coherence(self, assembled_text: str) -> CoherenceCheck:
        prompt = COHERENCE_CHECK_PROMPT.render(
            style_guide=self.style_guide,
            full_essay=assembled_text,
        )
        return await llm_engine._completion(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format=CoherenceCheck,
        )

    @staticmethod
    def assemble(sections_content: List[str]) -> str:
        return "\n\n".join(sections_content)

    @staticmethod
    def load_style_guide(path: str) -> str:
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
            return f"\nAUTHOR'S VOICE AND STYLE GUIDE:\n{content}\n\nYou MUST preserve this voice in all rewrites. Score LOW on voice fidelity if the output sounds like generic good writing rather than THIS specific writer.\n"
        return STYLE_GUIDE
