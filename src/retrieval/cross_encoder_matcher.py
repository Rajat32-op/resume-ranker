from sentence_transformers import CrossEncoder
import numpy as np

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
            model_name: str = "BAAI/bge-reranker-base"
    ):
        self.model = CrossEncoder(model_name)

    def score_all_dimensions(
            self,
            dimensions: list[Dimension],
            candidate: Candidate
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
            pairs.append((query, candidate.raw_text))
            names.append(dimension.name)

        if not pairs:
            return {}

        # predict returns raw logits
        scores = self.model.predict(pairs)

        # If only one pair, scores might be a single float or a 0-d array
        if isinstance(scores, (float, np.float32, np.float64, np.ndarray)) and np.ndim(scores) == 0:
            scores = [scores]

        return {
            name: float(score)
            for name, score in zip(names, scores)
        }
