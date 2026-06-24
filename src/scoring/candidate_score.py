from dataclasses import dataclass

from src.scoring.source_score import DimensionScore


@dataclass
class CandidateScore:

    candidate_id: str

    dimension_scores: dict[str, DimensionScore]

    final_score: float
