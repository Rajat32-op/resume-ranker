# src/scoring/source_score.py

from dataclasses import dataclass


@dataclass
class SourceScore:

    embedding_score: float
    bm25_score: float
    structured_score: float

    value: float


class SourceScorer:

    SOURCE_WEIGHTS = {

        "RANKING_RETRIEVAL": {
            "embedding": 0.3,
            "bm25": 0.2,
            "structured": 0.5
        },

        "PRODUCTION_EXPERIENCE": {
            "embedding": 0.2,
            "bm25": 0.1,
            "structured": 0.7
        },

        "TECHNICAL_SKILLS": {
            "embedding": 0.3,
            "bm25": 0.5,
            "structured": 0.2
        },

        "EVALUATION_FRAMEWORKS": {
            "embedding": 0.2,
            "bm25": 0.3,
            "structured": 0.5
        },

        "PRODUCT_MINDSET": {
            "embedding": 0.7,
            "bm25": 0.0,
            "structured": 0.3
        },

        "LEADERSHIP": {
            "embedding": 0.8,
            "bm25": 0.0,
            "structured": 0.2
        },

        "DOMAIN_EXPERIENCE": {
            "embedding": 0.4,
            "bm25": 0.2,
            "structured": 0.4
        },

        "LOCATION": {
            "embedding": 1.0,
            "bm25": 0.0,
            "structured": 0.0
        },

        "BEHAVIORAL_SIGNALS": {
            "embedding": 1.0,
            "bm25": 0.0,
            "structured": 0.0
        }
    }

    def score_dimension(
            self,
            dimension_name: str,
            embedding_score: float,
            bm25_score: float,
            structured_score: float
    ) -> SourceScore:

        weights = self.SOURCE_WEIGHTS.get(
            dimension_name,
            {
                "embedding": 0.3,
                "bm25": 0.2,
                "structured": 0.5
            }
        )

        value = (

            weights["embedding"] * embedding_score

            +

            weights["bm25"] * bm25_score

            +

            weights["structured"] * structured_score

        )

        return SourceScore(
            embedding_score=embedding_score,
            bm25_score=bm25_score,
            structured_score=structured_score,
            value=value
        )

    def score_all_dimensions(
            self,
            embedding_scores: dict[str, float],
            bm25_scores: dict[str, float],
            structured_scores
    ) -> dict[str, SourceScore]:

        dimensions = (
            set(embedding_scores.keys())
            |
            set(bm25_scores.keys())
            |
            set(structured_scores.keys())
        )

        results = {}

        for dimension in dimensions:

            embedding_score = embedding_scores.get(
                dimension,
                0.0
            )

            bm25_score = bm25_scores.get(
                dimension,
                0.0
            )

            structured_result = structured_scores.get(
                dimension
            )

            structured_score = (
                structured_result.score
                if structured_result is not None
                else 0.0
            )

            results[dimension] = self.score_dimension(
                dimension_name=dimension,
                embedding_score=embedding_score,
                bm25_score=bm25_score,
                structured_score=structured_score
            )

        return results