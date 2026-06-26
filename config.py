from pathlib import Path


EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"

EMBEDDING_MODEL_DIR = Path("artifacts/models/bge-small-en-v1.5")
RERANKER_MODEL_DIR = Path("artifacts/models/bge-reranker-base")


def resolve_model_path(
	local_path: Path,
	model_id: str,
	allow_remote: bool = False,
) -> str:
	if local_path.exists():
		return str(local_path)
	if allow_remote:
		return model_id
	raise FileNotFoundError(
		f"Missing local model snapshot at {local_path}. Run `python preprocess.py --download-models` first."
	)
