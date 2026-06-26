from dataclasses import dataclass

from src.scoring.source_score import SourceScore
from src.scoring.confidence import ConfidenceScore


@dataclass
class DimensionScore:

    source_score: float

    cross_encoder_score: float

    confidence_score: float

    final_score: float


class DimensionScorer:

    CROSS_ENCODER_WEIGHTS = {

        "RANKING_RETRIEVAL": 0.75,

        "PRODUCTION_EXPERIENCE": 0.40,

        "TECHNICAL_SKILLS": 0.40,

        "EVALUATION_FRAMEWORKS": 0.3,

        "DOMAIN_EXPERIENCE": 0.1

    }

    def score_dimension(
            self,
            dimension_name: str,
            source_score: SourceScore,
            confidence_score: ConfidenceScore,
            cross_encoder_score: float = 0.0
    ) -> DimensionScore:

        ce_weight = self.CROSS_ENCODER_WEIGHTS.get(
            dimension_name,
            0.0
        )

        source_weight = (
            1.0 - ce_weight
        )

        # Stage 1 score adjusted by confidence
        stage1_score = (
            source_score.value
            *
            confidence_score.value
        )

        final_score = (

            source_weight
            *
            stage1_score

            +

            ce_weight
            *
            cross_encoder_score

        )

        return DimensionScore(
            source_score=source_score.value,
            cross_encoder_score=cross_encoder_score,
            confidence_score=confidence_score.value,
            final_score=final_score
        )

    def score_all_dimensions(
            self,
            source_scores: dict[str, SourceScore],
            confidence_scores,
            cross_encoder_scores: dict[str, float]
    ) -> dict[str, DimensionScore]:

        results = {}

        dimensions = set(
            source_scores.keys()
        )

        for dimension_name in dimensions:

            source_score = source_scores[
                dimension_name
            ]

            confidence_score = (
                confidence_scores[
                    dimension_name
                ]
            )

            cross_encoder_score = (
                cross_encoder_scores.get(
                    dimension_name,
                    0.0
                )
            )

            results[
                dimension_name
            ] = self.score_dimension(
                dimension_name=dimension_name,
                source_score=source_score,
                confidence_score=confidence_score,
                cross_encoder_score=cross_encoder_score
            )

        return results