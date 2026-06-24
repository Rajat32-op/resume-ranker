import json
import numpy as np

from src.ranker import Ranker

from src.scoring.confidence import (
    ConfidenceScorer
)

from src.scoring.dimension_score import (
    DimensionScorer
)

from src.scoring.candidate_score import (
    CandidateScore
)

from src.retrieval.cross_encoder_matcher import (
    CrossEncoderMatcher
)


TOP_K = 1000


def normalize_cross_encoder_scores(
        candidate_ce_scores
):

    dimensions = set()

    for scores in candidate_ce_scores.values():
        dimensions.update(scores.keys())

    for dimension in dimensions:

        values = np.array([

            candidate_ce_scores[candidate_id].get(
                dimension,
                0.0
            )

            for candidate_id in candidate_ce_scores

        ])

        min_val = values.min()
        max_val = values.max()

        denom = max_val - min_val

        if denom < 1e-8:
            denom = 1.0

        for candidate_id in candidate_ce_scores:

            raw = candidate_ce_scores[
                candidate_id
            ].get(
                dimension,
                0.0
            )

            candidate_ce_scores[
                candidate_id
            ][
                dimension
            ] = (
                raw - min_val
            ) / denom

    return candidate_ce_scores


def main():

    ranker = Ranker(
        jd_path="data/job_description.docx",
        candidates_path="data/candidates.jsonl"
    )

    stage1_results = ranker.rank_candidates()

    top_candidates = stage1_results[:TOP_K]

    candidate_lookup = {

        candidate.candidate_id: candidate

        for candidate in ranker.candidates
    }

    confidence_scorer = (
        ConfidenceScorer()
    )

    dimension_scorer = (
        DimensionScorer()
    )

    cross_encoder_matcher = (
        CrossEncoderMatcher()
    )

    # --------------------------------------------------
    # Cross encoder scoring
    # --------------------------------------------------

    candidate_ce_scores = {}

    for result in top_candidates:

        candidate = candidate_lookup[
            result.candidate_id
        ]

        ce_scores = (
            cross_encoder_matcher
            .score_all_dimensions(
                ranker.dimensions,
                candidate
            )
        )

        candidate_ce_scores[
            candidate.candidate_id
        ] = ce_scores

    candidate_ce_scores = (
        normalize_cross_encoder_scores(
            candidate_ce_scores
        )
    )

    # --------------------------------------------------
    # Final scoring
    # --------------------------------------------------

    final_results = []

    for result in top_candidates:

        # Extract only SourceScore objects from dimension_scores
        current_source_scores = {
            dim_name: dim_score.source_score
            for dim_name, dim_score in result.dimension_scores.items()
        }

        confidence_scores = (
            confidence_scorer
            .score_all_dimensions(
                current_source_scores
            )
        )

        dimension_scores = (
            dimension_scorer
            .score_all_dimensions(
                source_scores=current_source_scores,
                confidence_scores=confidence_scores,
                cross_encoder_scores=
                candidate_ce_scores[
                    result.candidate_id
                ]
            )
        )

        final_score = 0.0

        for dimension in ranker.dimensions:

            if dimension.name not in dimension_scores:
                continue

            final_score += (

                dimension.weight

                *

                dimension_scores[
                    dimension.name
                ].final_score

            )

        final_results.append(

            CandidateScore(
                candidate_id=result.candidate_id,
                dimension_scores=result.dimension_scores,
                final_score=final_score
            )

        )

    final_results.sort(
        key=lambda x: x.final_score,
        reverse=True
    )

    print("\nTOP 100\n")

    for result in final_results[:100]:

        print(
            result.candidate_id,
            round(
                result.final_score,
                4
            )
        )


if __name__ == "__main__":
    main()