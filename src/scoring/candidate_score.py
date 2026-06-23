from dataclasses import dataclass

from src.scoring.source_score import SourceScore


@dataclass
class CandidateScore:

    candidate_id: str

    dimension_scores: dict[str, SourceScore]

    final_score: float