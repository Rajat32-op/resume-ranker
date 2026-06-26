from dataclasses import dataclass
from pathlib import Path
import re
import pickle

import numpy as np

from rank_bm25 import BM25Okapi

from src.retrieval.structured_matcher import StructuredMatcher
from src.parser.candidate_parser import load_candidates


STRUCTURED_DIMENSION_ORDER = [
    "RANKING_RETRIEVAL",
    "PRODUCTION_EXPERIENCE",
    "EVALUATION_FRAMEWORKS",
    "PRODUCT_MINDSET",
]


def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.split()


@dataclass
class PoolCandidateCache:
    candidate_ids: np.ndarray
    raw_texts: np.ndarray
    lower_texts: np.ndarray
    structured_scores: np.ndarray
    tokenized_corpus: np.ndarray | None = None


BM25_INDEX_PATH = "artifacts/pool_bm25.pkl"


def build_pool_candidate_cache(
    input_path: str,
    output_path: str,
    file_format: str | None = None,
    limit: int | None = None,
) -> PoolCandidateCache:
    candidates = load_candidates(
        input_path,
        file_format=file_format,
        limit=limit,
    )

    return build_pool_candidate_cache_from_candidates(candidates, output_path)


def build_pool_candidate_cache_from_candidates(
    candidates,
    output_path: str,
) -> PoolCandidateCache:

    structured_matcher = StructuredMatcher()

    candidate_ids = []
    raw_texts = []
    lower_texts = []
    tokenized_corpus = []
    structured_rows = []

    for candidate in candidates:
        candidate_ids.append(candidate.candidate_id)
        raw_texts.append(candidate.raw_text)
        lower_texts.append(candidate.lower_text)
        tokenized_corpus.append(tokenize_text(candidate.raw_text))

        structured_values = structured_matcher.score_all_dimensions_values(candidate)
        structured_rows.append([
            structured_values.get(dimension_name, 0.0)
            for dimension_name in STRUCTURED_DIMENSION_ORDER
        ])

    cache = PoolCandidateCache(
        candidate_ids=np.array(candidate_ids, dtype=object),
        raw_texts=np.array(raw_texts, dtype=object),
        lower_texts=np.array(lower_texts, dtype=object),
        tokenized_corpus=np.array(tokenized_corpus, dtype=object),
        structured_scores=np.array(structured_rows, dtype=np.float32),
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        candidate_ids=cache.candidate_ids,
        raw_texts=cache.raw_texts,
        lower_texts=cache.lower_texts,
        structured_scores=cache.structured_scores,
        tokenized_corpus=cache.tokenized_corpus,
    )

    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)

    return cache


def load_pool_candidate_cache(path: str) -> PoolCandidateCache:
    data = np.load(path, allow_pickle=True)
    return PoolCandidateCache(
        candidate_ids=data["candidate_ids"],
        raw_texts=data["raw_texts"],
        lower_texts=data["lower_texts"],
        structured_scores=data["structured_scores"],
        tokenized_corpus=data.get("tokenized_corpus", None),
    )
