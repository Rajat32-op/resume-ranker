import argparse
import os
import sys
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parser.candidate_parser import load_candidates


MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 256


def parse_args():

    parser = argparse.ArgumentParser(
        description="Build candidate embeddings for the ranking pipeline"
    )

    parser.add_argument(
        "--input",
        default="data/candidates.jsonl",
        help="Path to the candidate file (.json or .jsonl)"
    )

    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default=None,
        help="Explicit input format when it cannot be inferred from the file extension"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of candidates to embed"
    )

    parser.add_argument(
        "--output-embeddings",
        default="artifacts/candidate_embeddings.npy",
        help="Output path for the embedding matrix"
    )

    parser.add_argument(
        "--output-ids",
        default="artifacts/candidate_ids.npy",
        help="Output path for the candidate id array"
    )

    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="SentenceTransformer model name"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for embedding generation"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    os.makedirs(
        Path(args.output_embeddings).parent,
        exist_ok=True
    )

    print(
        "Loading candidates..."
    )

    candidates = load_candidates(
        args.input,
        file_format=args.format,
        limit=args.limit
    )

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    candidate_texts = [candidate.raw_text for candidate in candidates]

    print(
        f"Loaded {len(candidate_ids)} candidates"
    )

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        args.model_name
    )

    print(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(
        np.float16
    )

    print(
        "Saving embeddings..."
    )

    np.save(
        args.output_embeddings,
        embeddings
    )

    np.save(
        args.output_ids,
        np.array(
            candidate_ids,
            dtype=object
        )
    )

    print(
        f"Saved embeddings: "
        f"{args.output_embeddings}"
    )

    print(
        f"Saved ids: "
        f"{args.output_ids}"
    )

    print(
        f"Shape: {embeddings.shape}"
    )


if __name__ == "__main__":
    main()