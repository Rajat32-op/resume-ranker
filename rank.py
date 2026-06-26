import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from src.ranker import Ranker

DEFAULT_JD_PATH = "data/job_description.docx"
DEFAULT_SAMPLE_CANDIDATES_PATH = "data/sample_candidates.json"
DEFAULT_POOL_CANDIDATES_PATH = "data/candidates.jsonl"
DEFAULT_EMBEDDINGS_PATH = "artifacts/candidate_embeddings.npy"
DEFAULT_IDS_PATH = "artifacts/candidate_ids.npy"
DEFAULT_OUTPUT_DIR = "outputs"
FINAL_OUTPUT_TOP_N = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the resume ranking pipeline"
    )

    parser.add_argument(
        "--candidate-mode",
        choices=["sample", "pool"],
        default="sample",
        help="Sample mode uses <=100 uploaded candidates and computes embeddings live. Pool mode uses the 100k candidate pool and precomputed embeddings."
    )

    parser.add_argument(
        "--jd-path",
        default=DEFAULT_JD_PATH,
        help="Path to the job description docx"
    )

    parser.add_argument(
        "--candidates-path",
        default=None,
        help="Path to the candidate file. Defaults to sample_candidates.json in sample mode or candidates.jsonl in pool mode"
    )

    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write final results as JSON"
    )

    return parser.parse_args()


def resolve_defaults(args):
    if args.candidates_path is None:
        if args.candidate_mode == "pool":
            args.candidates_path = DEFAULT_POOL_CANDIDATES_PATH
        else:
            args.candidates_path = DEFAULT_SAMPLE_CANDIDATES_PATH

    return args


def serialize_results(results):
    serialized = []

    for result in results:
        serialized.append(
            {
                "candidate_id": result.candidate_id,
                "final_score": result.final_score,
                "dimension_scores": {
                    dimension_name: {
                        "source_score": dimension_score.source_score.value if dimension_score.source_score else 0.0,
                        "embedding_score": dimension_score.source_score.embedding_score if dimension_score.source_score else 0.0,
                        "bm25_score": dimension_score.source_score.bm25_score if dimension_score.source_score else 0.0,
                        "structured_score": dimension_score.source_score.structured_score if dimension_score.source_score else 0.0,
                        "cross_encoder_score": dimension_score.cross_encoder_score,
                        "confidence_score": getattr(dimension_score, "confidence_score", 0.0),
                        "final_score": dimension_score.final_score,
                    }
                    for dimension_name, dimension_score in result.dimension_scores.items()
                },
            }
        )

    return serialized


def main():
    args = resolve_defaults(parse_args())
    start_time = perf_counter()

    ranker = Ranker(
        jd_path=args.jd_path,
        candidates_path=args.candidates_path,
        use_precomputed_embeddings=args.candidate_mode == "pool",
        embeddings_path=DEFAULT_EMBEDDINGS_PATH,
        ids_path=DEFAULT_IDS_PATH,
    )

    results = ranker.rank_candidates()

    print(f"\nTOP {FINAL_OUTPUT_TOP_N}\n")

    for result in results[: FINAL_OUTPUT_TOP_N]:
        print(result.candidate_id, round(result.final_score, 4))

    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"ranking_results_{args.candidate_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("candidate_id,rank,score,reasoning\n")
        cnt=1
        for result in results[: FINAL_OUTPUT_TOP_N]:
            f.write(f"{result.candidate_id},{cnt},{result.final_score:.6f},{result.reason}\n")
            cnt += 1

    print(f"Saved CSV output to: {csv_path}")

    elapsed_seconds = perf_counter() - start_time
    print(f"Total time: {elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()
