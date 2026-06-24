from dataclasses import dataclass

import numpy as np

from src.scoring.source_score import SourceScore


@dataclass
class ConfidenceScore:

    agreement: float

    coverage: float

    value: float


class ConfidenceScorer:

    TOTAL_SOURCES = 3

    def score_dimension(
            self,
            source_score: SourceScore
    ) -> ConfidenceScore:

        scores = []

        if source_score.embedding_score > 0:
            scores.append(
                source_score.embedding_score
            )

        if source_score.bm25_score > 0:
            scores.append(
                source_score.bm25_score
            )

        if source_score.structured_score > 0:
            scores.append(
                source_score.structured_score
            )

        if len(scores) == 0:

            return ConfidenceScore(
                agreement=0.0,
                coverage=0.0,
                value=0.0
            )

        coverage = (
            len(scores)
            /
            self.TOTAL_SOURCES
        )

        std = np.std(scores)

        agreement = np.exp(
            -2.0 * std
        )

        value = (
            agreement
            *
            np.sqrt(
                coverage
            )
        )

        return ConfidenceScore(
            agreement=float(agreement),
            coverage=float(coverage),
            value=float(value)
        )

    def score_all_dimensions(
            self,
            source_scores: dict[str, SourceScore]
    ) -> dict[str, ConfidenceScore]:

        results = {}

        for (
                dimension_name,
                source_score
        ) in source_scores.items():

            results[
                dimension_name
            ] = self.score_dimension(
                source_score
            )

        return results