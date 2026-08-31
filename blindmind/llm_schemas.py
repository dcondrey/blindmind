from pydantic import BaseModel, Field


class MutationOutput(BaseModel):
    title: str = Field(..., description="A punchy title for the new concept")
    domain: str = Field(..., description="The primary domain of the new concept")
    description: str = Field(..., description="Detailed description of the concept and its mechanics")
    justification: str = Field(..., description="Why this combination/mutation results in a novel idea")


class CriticScore(BaseModel):
    conceptual_novelty: int = Field(..., ge=1, le=10, description="How unique is this combination?")
    feasibility: int = Field(..., ge=1, le=10, description="Is it logically/physically sound?")
    utility: int = Field(..., ge=1, le=10, description="Does it solve a real-world friction point?")
    semantic_jump: int = Field(
        ..., ge=1, le=10, description="How far is this from the parent ideas? (1=Incremental, 10=Radical Shift)"
    )
    prior_art_overlap: int = Field(
        default=1,
        ge=1,
        le=10,
        description="How much does this overlap with known existing work? (1=Completely Novel, 10=Already Exists)",
    )
    implementation_path: str = Field(
        default="", description="Brief sketch of how this could be built or realized in the real world"
    )
    fatal_flaws: list[str] = Field(default_factory=list, description="List of immediate risks or logical gaps")
    rationale: str = Field(..., description="Brief explanation for the scores")
    evolutionary_directive: str = Field(..., description="A short instruction for the next generation on how to adapt")

    @property
    def composite_score(self) -> float:
        # Self-rated novelty is the least reliable signal the critic produces (it
        # over-rates recombinations). Weight feasibility, the most falsifiable vector,
        # more heavily, and weight the prior-art penalty on par with novelty's own
        # weight instead of well below it, so "this already exists" can actually
        # cancel out an inflated novelty score.
        novelty_bonus = max(0, (10 - self.prior_art_overlap)) * 0.25
        raw = (self.feasibility * 1.5 + self.conceptual_novelty + self.utility + (self.semantic_jump * 0.5)) / 5.0
        # A candidate the critic itself flagged as broken should rarely clear a normal
        # (~7.0) retention threshold, but this is a penalty, not a hard gate: BlindMind
        # is also run with a deliberately low threshold for raw, human-curated divergence
        # (see headless_evolve.py), and a hard veto there would silently return zero
        # survivors instead of the flagged-but-still-worth-seeing candidates that mode
        # depends on. Scale with how many flaws were listed, capped so one nitpick
        # doesn't obliterate an otherwise-strong candidate.
        flaw_penalty = min(len(self.fatal_flaws), 3) * 2.5
        return raw + novelty_bonus - flaw_penalty
