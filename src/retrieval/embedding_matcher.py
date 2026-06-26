# src/retrieval/embedding_matcher.py

from sentence_transformers import SentenceTransformer
import numpy as np

from src.dimensions.dimension_schema import Dimension
from src.parser.schema import Candidate


class EmbeddingMatcher:

    def __init__(
            self,
            use_precomputed: bool = True,
            embeddings_path: str = "artifacts/candidate_embeddings.npy",
            ids_path: str = "artifacts/candidate_ids.npy",
            model_name: str = "BAAI/bge-small-en-v1.5"
    ):

        self.use_precomputed = use_precomputed

        self.model = SentenceTransformer(
            model_name
        )

        self.dimension_embeddings = {}

        if self.use_precomputed:

            self.candidate_embeddings = np.load(
                embeddings_path
            )

            self.candidate_ids = np.load(
                ids_path,
                allow_pickle=True
            )

            self.candidate_id_to_row = {

                candidate_id: idx

                for idx, candidate_id in enumerate(
                    self.candidate_ids
                )

            }

    def build_dimension_text(
            self,
            dimension: Dimension
    ) -> str:

        anchor_text = "\n".join(
            dimension.anchors
        )

        paragraph_text = "\n".join(
            dimension.jd_paragraphs
        )

        return (
            anchor_text
            + "\n"
            + paragraph_text
        )

    def build_dimension_embeddings(
            self,
            dimensions: list[Dimension]
    ):

        texts = [

            self.build_dimension_text(
                dimension
            )

            for dimension in dimensions

        ]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        self.dimension_embeddings = {

            dimension.name: embedding

            for dimension, embedding in zip(
                dimensions,
                embeddings
            )

        }

    def get_candidate_embedding(
            self,
            candidate: Candidate
    ) -> np.ndarray:

        if self.use_precomputed:
            row = self.candidate_id_to_row[candidate.candidate_id]
            return self.candidate_embeddings[row]

        return self.model.encode(
            candidate.raw_text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def get_candidate_embeddings_matrix(
            self,
            candidates: list[Candidate]
    ) -> np.ndarray:

        if self.use_precomputed:
            rows = [
                self.candidate_id_to_row[candidate.candidate_id]
                for candidate in candidates
            ]
            return self.candidate_embeddings[rows]

        return self.model.encode(
            [candidate.raw_text for candidate in candidates],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def score_candidate(
            self,
            candidate: Candidate
    ) -> dict[str, float]:

        candidate_embedding = self.get_candidate_embedding(candidate)
        scores = {}

        for dimension_name, dimension_embedding in self.dimension_embeddings.items():
            score = np.dot(dimension_embedding, candidate_embedding)
            scores[dimension_name] = float(score)

        return scores

    def score_candidates_bulk(
            self,
            dimensions: list[Dimension],
            candidates: list[Candidate]
    ) -> list[dict[str, float]]:

        if not self.dimension_embeddings:
            self.build_dimension_embeddings(dimensions)

        candidate_embeddings = self.get_candidate_embeddings_matrix(candidates)
        dimension_matrix = np.stack(
            [self.dimension_embeddings[dimension.name] for dimension in dimensions],
            axis=0
        )

        scores_matrix = candidate_embeddings @ dimension_matrix.T

        return [
            {
                dimension.name: float(score)
                for dimension, score in zip(dimensions, candidate_scores)
            }
            for candidate_scores in scores_matrix
        ]

    def score_all_dimensions(
            self,
            dimensions: list[Dimension],
            candidate: Candidate
    ) -> dict[str, float]:

        if not self.dimension_embeddings:
            self.build_dimension_embeddings(dimensions)

        return self.score_candidate(candidate)
