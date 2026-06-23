import re

from rank_bm25 import BM25Okapi

from src.dimensions.dimension_schema import Dimension
from src.parser.schema import Candidate


class BM25Matcher:

    def __init__(
            self,
            candidates: list[Candidate]
    ):

        self.candidates = candidates

        self.candidate_ids = [
            candidate.candidate_id
            for candidate in candidates
        ]

        self.corpus = [
            self.tokenize(
                candidate.raw_text
            )
            for candidate in candidates
        ]

        self.bm25 = BM25Okapi(
            self.corpus
        )

    def tokenize(
            self,
            text: str
    ) -> list[str]:

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9 ]",
            " ",
            text
        )

        return text.split()

    def build_dimension_query(
            self,
            dimension: Dimension
    ) -> list[str]:

        text = "\n".join(
            dimension.anchors
        )

        return self.tokenize(
            text
        )

    def score_dimension(
            self,
            dimension: Dimension
    ) -> dict[str, float]:

        query = self.build_dimension_query(
            dimension
        )

        scores = self.bm25.get_scores(
            query
        )

        max_score = max(scores)

        if max_score > 0:
            scores = scores / max_score

        return {

            candidate_id: float(score)

            for candidate_id, score in zip(
                self.candidate_ids,
                scores
            )

        }

    def score_all_dimensions(
            self,
            dimensions: list[Dimension]
    ) -> dict[str, dict[str, float]]:

        results = {}

        for dimension in dimensions:

            results[
                dimension.name
            ] = self.score_dimension(
                dimension
            )

        return results