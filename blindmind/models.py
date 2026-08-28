from datetime import datetime, timezone, UTC
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4
from sqlmodel import Field, Relationship, SQLModel


class MutationType(str, Enum):
    CROSSOVER = "CROSSOVER"
    POINT_MUTATION = "POINT_MUTATION"
    INVERSION = "INVERSION"
    WILDCARD = "WILDCARD"
    REWRITE = "REWRITE"
    TIGHTEN = "TIGHTEN"
    AMPLIFY = "AMPLIFY"


class RunStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Lineage(SQLModel, table=True):
    child_id: UUID = Field(foreign_key="concept.id", primary_key=True)
    parent_id: UUID = Field(foreign_key="concept.id", primary_key=True)
    mutation_type: MutationType = Field(default=MutationType.CROSSOVER)


class Concept(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project: str = Field(default="default", index=True)
    domain: str
    title: str
    description: str
    generation: int = Field(default=0, index=True)
    fitness_score: Optional[float] = None
    tags: Optional[str] = Field(default=None, description="Comma-separated tags for organization")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def tag_list(self) -> List[str]:
        if not self.tags:
            return []
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    @property
    def short_id(self) -> str:
        return str(self.id)[:8]


class EvolutionRun(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project: str = Field(default="default", index=True)
    status: RunStatus = Field(default=RunStatus.IN_PROGRESS)
    current_generation: int = Field(default=0)
    total_generations: int = Field(default=1)
    population_size: int = Field(default=5)
    latest_directive: Optional[str] = Field(default="Focus on maximizing combinatorial novelty and logical consistency.")
    config_snapshot: Optional[str] = None
    concepts_generated: int = Field(default=0)
    concepts_retained: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
