from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.dimensions.dimension_schema import Dimension
from src.parser.schema import Candidate


class EmbeddingMatcher:

    def __init__(
            self,
            model_name: str = "BAAI/bge-small-en-v1.5"
    ):

        self.model = SentenceTransformer(model_name)

    def build_dimension_text(
            self,
            dimension: Dimension
    ) -> str:
        """
        Convert a dimension into text to embed.
        """

        anchor_text = "\n".join(dimension.anchors)

        paragraph_text = "\n".join(dimension.jd_paragraphs)

        return anchor_text + "\n" + paragraph_text

    def score_dimension(
            self,
            dimension: Dimension,
            candidate: Candidate
    ) -> float:
        """
        Compute semantic similarity between a dimension
        and a candidate.
        """

        dimension_text = self.build_dimension_text(
            dimension
        )

        candidate_text = candidate.raw_text

        embeddings = self.model.encode(
            [dimension_text, candidate_text],
            normalize_embeddings=True
        )

        dimension_embedding = embeddings[0]
        candidate_embedding = embeddings[1]

        score = cosine_similarity(
            dimension_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0][0]

        return float(score)

    def score_all_dimensions(
            self,
            dimensions: list[Dimension],
            candidate: Candidate
    ) -> dict[str, float]:

        scores = {}

        for dimension in dimensions:

            scores[dimension.name] = (
                self.score_dimension(
                    dimension,
                    candidate
                )
            )

        return scores