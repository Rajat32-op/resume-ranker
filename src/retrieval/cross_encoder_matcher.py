from sentence_transformers import CrossEncoder
import numpy as np

from config import RERANKER_MODEL_DIR, RERANKER_MODEL_ID, resolve_model_path
from src.dimensions.dimension_schema import Dimension
from src.parser.schema import Candidate


class CrossEncoderMatcher:

    DIMENSION_QUERIES = {
        "TECHNICAL_SKILLS": "Core technical skills, programming languages, and tools required for the role.",
        "RANKING_RETRIEVAL": "Experience with ranking algorithms, recommendation systems, or information retrieval.",
        "PRODUCTION_EXPERIENCE": "Experience building and maintaining scalable production-level software systems.",
        "EVALUATION_FRAMEWORKS": "Expertise in designing and using frameworks to evaluate model performance and quality.",
        "DOMAIN_EXPERIENCE": "Relevant industry or vertical-specific knowledge and experience."
    }

    CE_DIMENSIONS = set(DIMENSION_QUERIES.keys())

    def __init__(
            self,
            model_name: str = RERANKER_MODEL_ID,
            batch_size: int = 128
    ):
        self.model = CrossEncoder(resolve_model_path(RERANKER_MODEL_DIR, model_name, allow_remote=False))
        self.batch_size = batch_size

    def _candidate_text(self, candidate: Candidate | str) -> str:
        return candidate if isinstance(candidate, str) else candidate.raw_text

    def score_all_dimensions(
            self,
            dimensions: list[Dimension],
            candidate: Candidate | str
    ) -> dict[str, float]:
        """
        Compute cross-encoder scores for relevant dimensions.
        Returns raw logits without normalization.
        """
        pairs = []
        names = []

        for dimension in dimensions:
            if dimension.name not in self.CE_DIMENSIONS:
                continue

            query = self.DIMENSION_QUERIES[dimension.name]
            pairs.append((query, self._candidate_text(candidate)))
            names.append(dimension.name)

        if not pairs:
            return {}

        scores = self.model.predict(pairs)

        if isinstance(scores, (float, np.float32, np.float64, np.ndarray)) and np.ndim(scores) == 0:
            scores = [scores]

        return {
            name: float(score)
            for name, score in zip(names, scores)
        }

    def score_dimensions_for_candidates(
            self,
            dimensions: list[Dimension],
            candidates: list[Candidate | str]
    ) -> list[dict[str, float]]:
        """
        Batch cross-encoder scoring by dimension to minimize predict calls.
        """
        if not candidates:
            return []

        scores_by_dimension: dict[str, list[float]] = {
            dimension.name: [0.0] * len(candidates)
            for dimension in dimensions
            if dimension.name in self.CE_DIMENSIONS
        }

        for dimension in dimensions:
            if dimension.name not in self.CE_DIMENSIONS:
                continue

            query = self.DIMENSION_QUERIES[dimension.name]
            pairs = [(query, self._candidate_text(candidate)) for candidate in candidates]
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            if isinstance(scores, (float, np.float32, np.float64)):
                scores = [scores]

            scores_by_dimension[dimension.name] = [float(score) for score in scores]

        return [
            {
                dimension_name: scores_by_dimension[dimension_name][index]
                for dimension_name in scores_by_dimension
            }
            for index in range(len(candidates))
        ]
