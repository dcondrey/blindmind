from pydantic import BaseModel, Field
from typing import List

class SectionParse(BaseModel):
    sections: List["SectionInfo"] = Field(..., description="The essay split into rhetorical sections")

class SectionInfo(BaseModel):
    title: str = Field(..., description="A short label for this section (e.g., 'The morning ritual', 'The turn')")
    function: str = Field(..., description="Rhetorical function: setup, escalation, evidence, turn, reflection, climax, close")
    content: str = Field(..., description="The full text of this section, preserved exactly")

class SectionRewrite(BaseModel):
    content: str = Field(..., description="The rewritten section text")
    approach: str = Field(..., description="Brief description of what was changed and why")

class EssayCriticScore(BaseModel):
    voice_fidelity: int = Field(..., ge=1, le=10, description="Does this sound like the author? Checks for second-person address, dark humor, visceral physicality, anti-inspirational stance, conversational authority")
    emotional_impact: int = Field(..., ge=1, le=10, description="Does this hit? Does the reader feel something unexpected? Does it build or release tension appropriately?")
    precision: int = Field(..., ge=1, le=10, description="Is every word earning its place? No filler, no hedge words, no passive voice where active would cut deeper?")
    coherence: int = Field(..., ge=1, le=10, description="Does this connect to surrounding sections? Does the essay still read as one continuous experience?")
    originality: int = Field(..., ge=1, le=10, description="Does this avoid cliche? Does it find a new way to say what needs saying?")
    strengths: List[str] = Field(default_factory=list, description="What works well in this version")
    weaknesses: List[str] = Field(default_factory=list, description="What could be stronger")
    rationale: str = Field(..., description="Brief explanation of scores")

    @property
    def composite_score(self) -> float:
        return (
            (self.voice_fidelity * 2) +
            (self.emotional_impact * 2) +
            self.precision +
            self.coherence +
            self.originality
        ) / 7.0

class CoherenceCheck(BaseModel):
    seams: List[str] = Field(default_factory=list, description="Places where tone or voice shifts awkwardly between sections")
    broken_callbacks: List[str] = Field(default_factory=list, description="References to earlier material that no longer connect")
    momentum_breaks: List[str] = Field(default_factory=list, description="Places where the essay loses forward energy")
    overall_coherence: int = Field(..., ge=1, le=10, description="How well does the assembled essay read as one piece?")
    suggestions: List[str] = Field(default_factory=list, description="Specific fixes for the issues found")
