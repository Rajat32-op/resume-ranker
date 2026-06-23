import json

from src.parser.jd_parser import parse_jd
from src.parser.candidate_parser import parse_candidate

from src.dimensions.anchor_builder import (
    build_dimension_templates
)
from src.dimensions.dimension_mapper import (
    map_jd_to_dimensions
)
from src.dimensions.weight_generator import (
    generate_weights
)

from src.retrieval.embedding_matcher import (
    EmbeddingMatcher
)
from src.retrieval.bm25_matcher import (
    BM25Matcher
)
from src.retrieval.structured_matcher import (
    StructuredMatcher
)

from src.scoring.source_score import (
    SourceScorer
)
from src.scoring.candidate_score import (
    CandidateScore
)


class Ranker:

    def __init__(
            self,
            jd_path: str,
            candidates_path: str
    ):

        # ---------- JD ----------
        self.jd = parse_jd(jd_path)

        self.dimensions = build_dimension_templates()

        self.dimensions = map_jd_to_dimensions(
            self.jd,
            self.dimensions
        )

        self.dimensions = generate_weights(
            self.dimensions
        )

        # ---------- candidates ----------
        with open(candidates_path) as f:
            candidate_jsons = json.load(f)

        self.candidates = [

            parse_candidate(
                candidate_json
            )

            for candidate_json in candidate_jsons
        ]

        # ---------- matchers ----------
        self.embedding_matcher = EmbeddingMatcher()

        self.bm25_matcher = BM25Matcher(
            self.candidates
        )

        self.structured_matcher = StructuredMatcher()

        self.source_scorer = SourceScorer()

    def score_candidate(
            self,
            candidate
    ) -> CandidateScore:

        # ---------- embedding ----------
        embedding_scores = (
            self.embedding_matcher.score_all_dimensions(
                self.dimensions,
                candidate
            )
        )

        # ---------- bm25 ----------
        bm25_dimensions = [

            dim

            for dim in self.dimensions

            if dim.name in {

                "TECHNICAL_SKILLS",
                "RANKING_RETRIEVAL",
                "PRODUCTION_EXPERIENCE",
                "EVALUATION_FRAMEWORKS",
                "DOMAIN_EXPERIENCE"

            }

        ]

        bm25_results = (
            self.bm25_matcher.score_all_dimensions(
                bm25_dimensions
            )
        )

        bm25_scores = {

            dimension_name:
                bm25_results[
                    dimension_name
                ][
                    candidate.candidate_id
                ]

            for dimension_name in bm25_results

        }

        # ---------- structured ----------
        structured_scores = (
            self.structured_matcher.score_all_dimensions(
                candidate
            )
        )

        # ---------- source scores ----------
        source_scores = (
            self.source_scorer.score_all_dimensions(
                embedding_scores,
                bm25_scores,
                structured_scores
            )
        )

        # ---------- final score ----------
        final_score = 0

        for dimension in self.dimensions:

            if dimension.name not in source_scores:
                continue

            final_score += (
                dimension.weight
                *
                source_scores[
                    dimension.name
                ].value
            )

        return CandidateScore(
            candidate_id=candidate.candidate_id,
            dimension_scores=source_scores,
            final_score=final_score
        )

    def rank_candidates(
            self
    ) -> list[CandidateScore]:

        results = []

        for candidate in self.candidates:

            results.append(
                self.score_candidate(
                    candidate
                )
            )

        results.sort(
            key=lambda x: x.final_score,
            reverse=True
        )

        return results