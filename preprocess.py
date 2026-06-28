from pathlib import Path
import argparse

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

from config import (
    EMBEDDING_MODEL_DIR,
    EMBEDDING_MODEL_ID,
    RERANKER_MODEL_DIR,
    RERANKER_MODEL_ID,
)
from src.parser.candidate_parser import load_candidates
from src.parser.pool_cache import build_pool_candidate_cache_from_candidates

HF_REPO_ID = "Rajat3206/redrob-artifacts"
INPUT_PATH = "data/candidates.jsonl"
EMBEDDINGS_PATH = "artifacts/candidate_embeddings.npy"
IDS_PATH = "artifacts/candidate_ids.npy"
POOL_CACHE_PATH = "artifacts/pool_candidate_cache.npz"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download only the embedding and reranker models."
    )

    return parser.parse_args()

def download_model_snapshot(repo_id: str, local_dir: Path):
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )


def download_pool_artifacts():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading candidates.jsonl from Hugging Face...")

    hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename="candidates.jsonl",
        local_dir="data",
    )
    files = [
        "candidate_embeddings.npy",
        "candidate_ids.npy",
    ]

    for filename in files:
        print(f"Downloading {filename} from Hugging Face...")

        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir="artifacts",
        )

def download_model_artifacts():
    print("Downloading embedding model snapshot...")
    download_model_snapshot(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_DIR)

    print("Downloading reranker model snapshot...")
    download_model_snapshot(RERANKER_MODEL_ID, RERANKER_MODEL_DIR)


def main():

    args = parse_args()

    print("Downloading model artifacts...")
    download_model_artifacts()

    if args.download_models:
        print("Model download completed.")
        return

    print("Downloading pool artifacts...")
    download_pool_artifacts()

    print("Loading candidates...")
    candidates = load_candidates(
        INPUT_PATH,
        file_format="jsonl"
    )

    print(f"Loaded {len(candidates)} candidates")

    print("Building pool cache...")
    build_pool_candidate_cache_from_candidates(
        candidates,
        POOL_CACHE_PATH
    )

    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved ids: {IDS_PATH}")
    print(f"Saved pool cache: {POOL_CACHE_PATH}")


if __name__ == "__main__":
    main()
