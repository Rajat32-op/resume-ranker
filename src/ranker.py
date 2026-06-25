from typing import List

from src.parser.jd_parser import parse_jd
from src.parser.candidate_parser import load_candidates
from src.parser.schema import Candidate

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
            ids_path: str = "artifacts/candidate_ids.npy"
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
        self.embedding_matcher = EmbeddingMatcher(
            use_precomputed=use_precomputed_embeddings,
            embeddings_path=embeddings_path,
            ids_path=ids_path
        )

        self.bm25_matcher = BM25Matcher(
            self.candidates
        )

        self.structured_matcher = StructuredMatcher()
        
        self.ce_matcher = CrossEncoderMatcher()

        self.source_scorer = SourceScorer()

    def score_candidate_stage1(
            self,
            candidate,
            bm25_scores: dict[str, float]
    ) -> dict:
        """
        Compute Stage 1 scores.
        """
        # ---------- embedding ----------
        embedding_scores = (
            self.embedding_matcher.score_all_dimensions(
                self.dimensions,
                candidate
            )
        )

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

    def rank_candidates(
            self,
            top_k: int = 1000
    ) -> List:

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
        
        # 1. Run Stage 1 for all candidates
        stage1_results = []
        for candidate in self.candidates:
            bm25_scores = {
                dimension_name: bm25_results[dimension_name][candidate.candidate_id]
                for dimension_name in bm25_results
            }

            stage1_results.append(
                self.score_candidate_stage1(
                    candidate,
                    bm25_scores=bm25_scores
                )
            )
        
        # Sort by Stage 1 score and take top K
        stage1_results.sort(key=lambda x: x["stage1_final_score"], reverse=True)
        top_candidates_data = stage1_results[:top_k]
        
        # 2. Run Stage 2 (Cross-Encoder) for top candidates
        all_ce_scores = {dim: [] for dim in CrossEncoderMatcher.CE_DIMENSIONS}
        candidate_ce_results = []
        
        for data in top_candidates_data:
            candidate = data["candidate"]
            ce_scores = self.ce_matcher.score_all_dimensions(self.dimensions, candidate)
            candidate_ce_results.append(ce_scores)
            
            for dim, score in ce_scores.items():
                all_ce_scores[dim].append(score)
        
        # 3. Normalize CE scores dimension-wise to [0, 1]
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
        
        # 4. Combine scores and build final CandidateScore objects
        final_results = []
        for i, data in enumerate(top_candidates_data):
            candidate = data["candidate"]
            source_scores = data["dimension_scores"]
            
            dimension_scores = {}
            total_final_score = 0
            
            for dimension in self.dimensions:
                dim_name = dimension.name
                s1_score = source_scores.get(dim_name)
                
                # Default values if dimension not in matchers
                s1_val = s1_score.value if s1_score else 0.0
                ce_val = 0.0
                
                if dim_name in CrossEncoderMatcher.CE_DIMENSIONS:
                    # Get the normalized score for this candidate (at index i)
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
            
            final_results.append(
                CandidateScore(
                    candidate_id=candidate.candidate_id,
                    dimension_scores=dimension_scores,
                    final_score=total_final_score
                )
            )
            
        # Re-rank based on final total score
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results
