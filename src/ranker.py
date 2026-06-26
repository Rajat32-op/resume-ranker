import heapq
from concurrent.futures import ThreadPoolExecutor
import json
from typing import List
from unittest import result

from src.scoring.reason_generator import ReasonGenerator
from src.parser.jd_parser import parse_jd
from src.parser.candidate_parser import load_candidates,parse_candidate
from src.parser.schema import Candidate
from src.parser.pool_cache import (
    load_pool_candidate_cache,
    STRUCTURED_DIMENSION_ORDER,
    BM25_INDEX_PATH,
)

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
from src.retrieval.cross_encoder_matcher import (
    CrossEncoderMatcher
)

from src.scoring.source_score import (
    SourceScorer, DimensionScore
)
from src.scoring.candidate_score import (
    CandidateScore
)

from src.scoring.reason_generator import (
    ReasonGenerator
)

class Ranker:

    CE_COMBINATION_WEIGHTS = {
        "RANKING_RETRIEVAL": 0.7,
        "TECHNICAL_SKILLS": 0.5,
        "PRODUCTION_EXPERIENCE": 0.6,
        "EVALUATION_FRAMEWORKS": 0.5,
        "DOMAIN_EXPERIENCE": 0.3
    }

    def __init__(
            self,
            jd_path: str,
            candidates_path: str | None = None,
            candidates: list[Candidate] | None = None,
            candidate_file_format: str | None = None,
            candidate_limit: int | None = None,
            use_precomputed_embeddings: bool = False,
            embeddings_path: str = "artifacts/candidate_embeddings.npy",
            ids_path: str = "artifacts/candidate_ids.npy",
            preprocessed_pool_cache_path: str = "artifacts/pool_candidate_cache.npz"
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

        self.pool_cache = None
        self.pool_candidate_ids = None
        self.pool_raw_texts = None
        self.pool_structured_scores = None
        self._model_executor = ThreadPoolExecutor(max_workers=2)
        self._embedding_matcher = None
        self._ce_matcher = None
        self._embedding_matcher_future = None
        self._ce_matcher_future = None

        # ---------- candidates ----------
        if use_precomputed_embeddings:
            try:
                self.pool_cache = load_pool_candidate_cache(preprocessed_pool_cache_path)
                self.pool_candidate_ids = list(self.pool_cache.candidate_ids)
                self.pool_raw_texts = list(self.pool_cache.raw_texts)
                self.pool_structured_scores = self.pool_cache.structured_scores
                self.candidates = None
            except FileNotFoundError:
                self.pool_cache = None

        if self.pool_cache is None:
            if candidates is not None:
                self.candidates = candidates
            elif candidates_path is not None:
                self.candidates = load_candidates(
                    candidates_path,
                    file_format=candidate_file_format,
                    limit=candidate_limit
                )
            else:
                raise ValueError(
                    "Either candidates or candidates_path must be provided"
                )

        # ---------- matchers ----------
        self._embedding_matcher_future = self._model_executor.submit(
            self._build_embedding_matcher,
            use_precomputed_embeddings,
            embeddings_path,
            ids_path,
        )

        self._ce_matcher_future = self._model_executor.submit(
            self._build_cross_encoder_matcher
        )

        if self.pool_cache is not None:
            self.bm25_matcher = BM25Matcher.from_bm25_index(
                candidate_ids=self.pool_candidate_ids,
                bm25_path=BM25_INDEX_PATH,
            )
        else:
            self.bm25_matcher = BM25Matcher(
                self.candidates
            )

        self.structured_matcher = StructuredMatcher()

        self.source_scorer = SourceScorer()

    def _build_embedding_matcher(
            self,
            use_precomputed_embeddings: bool,
            embeddings_path: str,
            ids_path: str
    ) -> EmbeddingMatcher:
        return EmbeddingMatcher(
            use_precomputed=use_precomputed_embeddings,
            embeddings_path=embeddings_path,
            ids_path=ids_path,
        )

    def _build_cross_encoder_matcher(self) -> CrossEncoderMatcher:
        return CrossEncoderMatcher()

    def _get_embedding_matcher(self) -> EmbeddingMatcher:
        if self._embedding_matcher is None:
            self._embedding_matcher = self._embedding_matcher_future.result()
        return self._embedding_matcher

    def _get_ce_matcher(self) -> CrossEncoderMatcher:
        if self._ce_matcher is None:
            self._ce_matcher = self._ce_matcher_future.result()
        return self._ce_matcher

    def score_candidate_stage1(
            self,
            candidate,
            bm25_scores: dict[str, float],
            embedding_scores: dict[str, float]
    ) -> dict:
        """
        Compute Stage 1 scores.
        """
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

        # Calculate preliminary stage 1 final score for ranking
        stage1_final_score = 0
        for dimension in self.dimensions:
            if dimension.name in source_scores:
                stage1_final_score += dimension.weight * source_scores[dimension.name].value

        return {
            "candidate": candidate,
            "dimension_scores": source_scores,
            "stage1_final_score": stage1_final_score
        }

    def score_candidate_stage1_value(
            self,
            candidate,
            bm25_scores: dict[str, float],
            embedding_scores: dict[str, float]
    ) -> float:

        structured_scores = self.structured_matcher.score_all_dimensions_values(candidate)

        stage1_final_score = 0.0
        for dimension in self.dimensions:
            stage1_final_score += dimension.weight * self.source_scorer.score_dimension_value(
                dimension_name=dimension.name,
                embedding_score=embedding_scores.get(dimension.name, 0.0),
                bm25_score=bm25_scores.get(dimension.name, 0.0),
                structured_score=structured_scores.get(dimension.name, 0.0)
            )

        return stage1_final_score

    def rank_candidates_pool(
            self
    ) -> List[CandidateScore]:

        STAGE1_TOP_K = 100

        candidate_ids = list(self.pool_candidate_ids)
        raw_texts = list(self.pool_raw_texts)
        structured_rows = self.pool_structured_scores

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

        bm25_results = self.bm25_matcher.score_all_dimensions(
            bm25_dimensions
        )

        embedding_matcher = self._get_embedding_matcher()

        embedding_scores_list = embedding_matcher.score_candidates_bulk(
            self.dimensions,
            candidate_ids
        )

        stage1_results = []
        for index, candidate_id in enumerate(candidate_ids):
            bm25_scores = {
                dimension_name: bm25_results[dimension_name][candidate_id]
                for dimension_name in bm25_results
            }
            embedding_scores = embedding_scores_list[index]
            structured_scores = {
                dimension_name: float(value)
                for dimension_name, value in zip(
                    STRUCTURED_DIMENSION_ORDER,
                    structured_rows[index]
                )
            }

            stage1_final_score = 0.0
            for dimension in self.dimensions:
                stage1_final_score += dimension.weight * self.source_scorer.score_dimension_value(
                    dimension_name=dimension.name,
                    embedding_score=embedding_scores.get(dimension.name, 0.0),
                    bm25_score=bm25_scores.get(dimension.name, 0.0),
                    structured_score=structured_scores.get(dimension.name, 0.0)
                )

            stage1_results.append(
                {
                    "candidate_id": candidate_id,
                    "raw_text": raw_texts[index],
                    "embedding_scores": embedding_scores,
                    "bm25_scores": bm25_scores,
                    "structured_scores": structured_scores,
                    "stage1_final_score": stage1_final_score,
                }
            )

        top_candidates_data = heapq.nlargest(
            STAGE1_TOP_K,
            stage1_results,
            key=lambda x: x["stage1_final_score"]
        )

        top_ids = {data["candidate_id"] for data in top_candidates_data}

        candidate_lookup = {}

        with open("data/candidates.jsonl", "r") as f:

            for line in f:

                candidate_json = json.loads(line)

                if candidate_json["candidate_id"] not in top_ids:
                    continue

                candidate = parse_candidate(candidate_json)

                candidate_lookup[candidate.candidate_id] = candidate

                if len(candidate_lookup) == len(top_ids):
                    break

        all_ce_scores = {dim: [] for dim in CrossEncoderMatcher.CE_DIMENSIONS}
        ce_candidates = [data["raw_text"] for data in top_candidates_data]
        ce_matcher = self._get_ce_matcher()

        ce_scores_list = ce_matcher.score_dimensions_for_candidates(
            self.dimensions,
            ce_candidates
        )

        for data, ce_scores in zip(top_candidates_data, ce_scores_list):
            data["ce_scores"] = ce_scores
            for dim, score in ce_scores.items():
                all_ce_scores[dim].append(score)

        normalized_ce_scores = {}
        for dim, scores in all_ce_scores.items():
            if not scores:
                continue
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized_ce_scores[dim] = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_ce_scores[dim] = [0.0 for _ in scores]

        final_results = []
        for i, data in enumerate(top_candidates_data):
            dimension_scores = {}
            total_final_score = 0.0

            for dimension in self.dimensions:
                dim_name = dimension.name
                s1_score = self.source_scorer.score_dimension(
                    dimension_name=dim_name,
                    embedding_score=data["embedding_scores"].get(dim_name, 0.0),
                    bm25_score=data["bm25_scores"].get(dim_name, 0.0),
                    structured_score=data["structured_scores"].get(dim_name, 0.0)
                )
                s1_val = s1_score.value
                ce_val = 0.0

                if dim_name in CrossEncoderMatcher.CE_DIMENSIONS:
                    ce_val = normalized_ce_scores[dim_name][i]
                    ce_weight = self.CE_COMBINATION_WEIGHTS.get(dim_name, 0.5)
                    final_dim_score = (ce_weight * ce_val) + ((1 - ce_weight) * s1_val)
                else:
                    final_dim_score = s1_val

                dimension_scores[dim_name] = DimensionScore(
                    source_score=s1_score,
                    cross_encoder_score=ce_val,
                    final_score=final_dim_score
                )

                total_final_score += dimension.weight * final_dim_score
            reason_generator = ReasonGenerator()
            reason = reason_generator.generate(
                candidate=candidate_lookup[data["candidate_id"]],
                dimension_scores=dimension_scores
            )
            final_results.append(
                CandidateScore(
                    candidate_id=data["candidate_id"],
                    dimension_scores=dimension_scores,
                    final_score=total_final_score,
                    reason=reason
                )
            )

        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results

    def rank_candidates(
            self
    ) -> List[CandidateScore]:

        if self.pool_cache is not None:
            return self.rank_candidates_pool()

        embedding_matcher = self._get_embedding_matcher()

        STAGE1_TOP_K = 100

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

        bm25_results = self.bm25_matcher.score_all_dimensions(
            bm25_dimensions
        )

        embedding_scores_list = embedding_matcher.score_candidates_bulk(
            self.dimensions,
            self.candidates
        )

        embedding_scores_by_id = {
            candidate.candidate_id: embedding_scores
            for candidate, embedding_scores in zip(self.candidates, embedding_scores_list)
        }

        stage1_results = []
        for candidate, embedding_scores in zip(self.candidates, embedding_scores_list):
            bm25_scores = {
                dimension_name: bm25_results[dimension_name][candidate.candidate_id]
                for dimension_name in bm25_results
            }

            stage1_results.append(
                {
                    "candidate": candidate,
                    "stage1_final_score": self.score_candidate_stage1_value(
                        candidate,
                        bm25_scores=bm25_scores,
                        embedding_scores=embedding_scores
                    )
                }
            )

        top_candidates_data = heapq.nlargest(
            STAGE1_TOP_K,
            stage1_results,
            key=lambda x: x["stage1_final_score"]
        )

        ce_matcher = self._get_ce_matcher()

        for data in top_candidates_data:
            candidate = data["candidate"]
            bm25_scores = {
                dimension_name: bm25_results[dimension_name][candidate.candidate_id]
                for dimension_name in bm25_results
            }
            embedding_scores = embedding_scores_by_id[candidate.candidate_id]
            data.update(
                self.score_candidate_stage1(
                    candidate,
                    bm25_scores=bm25_scores,
                    embedding_scores=embedding_scores
                )
            )

        all_ce_scores = {dim: [] for dim in CrossEncoderMatcher.CE_DIMENSIONS}

        top_candidates = [data["candidate"] for data in top_candidates_data]
        ce_scores_list = ce_matcher.score_dimensions_for_candidates(
            self.dimensions,
            top_candidates
        )

        for data, ce_scores in zip(top_candidates_data, ce_scores_list):
            data["ce_scores"] = ce_scores
            for dim, score in ce_scores.items():
                all_ce_scores[dim].append(score)

        normalized_ce_scores = {}
        for dim, scores in all_ce_scores.items():
            if not scores:
                continue
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized_ce_scores[dim] = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_ce_scores[dim] = [0.0 for _ in scores]

        final_results = []
        for i, data in enumerate(top_candidates_data):
            candidate = data["candidate"]
            source_scores = data["dimension_scores"]

            dimension_scores = {}
            total_final_score = 0

            for dimension in self.dimensions:
                dim_name = dimension.name
                s1_score = source_scores.get(dim_name)
                s1_val = s1_score.value if s1_score else 0.0
                ce_val = 0.0

                if dim_name in CrossEncoderMatcher.CE_DIMENSIONS:
                    ce_val = normalized_ce_scores[dim_name][i]
                    ce_weight = self.CE_COMBINATION_WEIGHTS.get(dim_name, 0.5)
                    final_dim_score = (ce_weight * ce_val) + ((1 - ce_weight) * s1_val)
                else:
                    final_dim_score = s1_val

                dimension_scores[dim_name] = DimensionScore(
                    source_score=s1_score,
                    cross_encoder_score=ce_val,
                    final_score=final_dim_score
                )

                total_final_score += dimension.weight * final_dim_score
            reason_generator = ReasonGenerator()

            reason = reason_generator.generate(
                candidate=candidate,
                dimension_scores=dimension_scores
            )
            final_results.append(
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    dimension_scores=dimension_scores,
                    final_score=total_final_score,
                    reason=reason
                )
            )

        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results
