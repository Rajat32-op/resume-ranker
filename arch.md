# Architecture

## Overview

This repository implements a two-stage resume ranking pipeline around a single job description (JD) and a set of candidate profiles.

The system has two execution modes:

1. Sample mode
  - The user uploads a small JSON file with candidates.
  - The app and CLI cap the sample to 100 candidates internally.
  - Candidate embeddings are generated live from raw text.
  - The full ranking pipeline is then executed.

2. Pool mode
   - The user uploads the large candidate pool JSONL file.
   - Candidate embeddings are loaded from precomputed artifacts in `artifacts/`.
   - The ranking pipeline runs over the full pool without recomputing candidate embeddings.

The pipeline is split into these phases:

- JD parsing and dimension construction
- Candidate parsing and text normalization
- Stage 1 retrieval scoring
- Cross-encoder reranking on the Stage 1 shortlist
- Final score aggregation and output formatting

## Runtime Entry Points

### CLI

File: `rank.py`

Main responsibilities:

- Parse CLI arguments.
- Choose sample mode or pool mode.
- Instantiate `src.ranker.Ranker`.
- Execute ranking.
- Print the top 100 results.

Key functions:

- `parse_args()`
  - Defines the CLI interface.
  - The only user-visible mode switch is `--candidate-mode`.
- `resolve_defaults(args)`
  - Fills in default candidate file paths.
  - Keeps the reranking depth and output count fixed inside the program.
- `serialize_results(results)`
  - Converts ranking output to JSON.
- `main()`
  - Wires the CLI to the ranking engine.

Important constants:

- `STAGE1_TOP_K = 1000`
- `FINAL_OUTPUT_TOP_N = 100`

These are intentionally internal so the user does not control ranking depth from the CLI.

### Streamlit UI

File: `app.py`

Main responsibilities:

- Provide a simple upload interface for Hugging Face Spaces.
- Let the user choose sample mode or pool mode.
- Accept the JD upload and the candidate file upload.
- Run the same ranking pipeline used by the CLI.
- Display and download the top 100 results.

Key runtime flow:

- `st.file_uploader()` is used for the JD.
- `st.file_uploader()` is used for the candidate file.
- `load_candidates_from_records()` converts uploaded JSON or JSONL records into candidate objects.
- `Ranker(...)` executes the pipeline.
- A Pandas DataFrame shows the final ranking table.
- The UI does not expose tuning knobs for Stage 1 depth or final output count.

## Core Pipeline

### 1. JD parsing

File: `src/parser/jd_parser.py`

Function:

- `parse_jd(docx_path)`

What it does:

- Reads the JD `.docx` file.
- Extracts raw paragraphs.
- Parses structured fields such as title, company, location, employment type, and experience range.
- Produces a `JD` object with `raw_text` plus parsed metadata.

Output:

- `src/parser/jd_schema.JD`

### 2. Dimension construction and weighting

Files:

- `src/dimensions/anchor_builder.py`
- `src/dimensions/dimension_mapper.py`
- `src/dimensions/weight_generator.py`

This phase builds the scoring dimensions that the whole ranking system uses.

Functions:

- `build_dimension_templates()`
  - Creates the base dimension list and their anchor text.
- `map_jd_to_dimensions(jd, dimensions)`
  - Maps JD content into the dimension templates.
- `generate_weights(dimensions)`
  - Assigns final weights for each dimension.

These functions are called from `Ranker.__init__()`.

### 3. Candidate parsing

File: `src/parser/candidate_parser.py`

Functions:

- `parse_candidate(candidate_json)`
  - Parses one candidate dictionary into a `Candidate` object.
- `load_candidates_from_records(candidate_records)`
  - Converts a list of dictionaries into parsed candidates.
- `load_candidates(path, file_format=None, limit=None)`
  - Loads candidates from JSON or JSONL.
  - Supports optional limiting for sample mode.

Output:

- `src/parser/schema.Candidate`

Important behavior:

- The parser also constructs `candidate.raw_text`.
- That raw text is used by embedding, BM25, structured, and cross-encoder retrieval.

### 4. Stage 1 retrieval scoring

File: `src/ranker.py`

Main function:

- `Ranker.score_candidate_stage1(candidate, bm25_scores)`

This stage combines three source signals:

- Embedding similarity
- BM25 lexical similarity
- Structured heuristic matching

The stage uses these components:

#### 4.1 Embedding similarity

File: `src/retrieval/embedding_matcher.py`

Class: `EmbeddingMatcher`

Key functions:

- `__init__(use_precomputed=True, embeddings_path=..., ids_path=..., model_name=...)`
- `build_dimension_text(dimension)`
- `build_dimension_embeddings(dimensions)`
- `get_candidate_embedding(candidate)`
- `score_candidate(candidate)`
- `score_all_dimensions(dimensions, candidate)`

Behavior:

- In sample mode, candidate embeddings are generated live from `candidate.raw_text`.
- In pool mode, candidate embeddings are loaded once from `artifacts/candidate_embeddings.npy` and looked up by candidate id.
- Dimension embeddings are always generated from the JD dimensions.

This is the component that handles the precomputed embedding flag.

#### 4.2 BM25 retrieval

File: `src/retrieval/bm25_matcher.py`

Class: `BM25Matcher`

Key functions:

- `tokenize(text)`
- `build_dimension_query(dimension)`
- `score_dimension(dimension)`
- `score_all_dimensions(dimensions)`

Behavior:

- Builds a BM25 index over all candidates once.
- Scores each JD dimension query against the candidate corpus.
- Returns normalized lexical scores per candidate and per dimension.

Performance note:

- BM25 is computed once per ranking run, not once per candidate.
- This avoids the expensive repeated full-corpus BM25 pass.

#### 4.3 Structured heuristics

File: `src/retrieval/structured_matcher.py`

Class: `StructuredMatcher`

Key functions:

- `score_ranking_retrieval(candidate)`
- `score_production_experience(candidate)`
- `score_evaluation_frameworks(candidate)`
- `score_product_mindset(candidate)`
- `score_all_dimensions(candidate)`

Behavior:

- Uses simple keyword and rule-based checks on `candidate.raw_text` and profile metadata.
- Produces a `MatchResult` for each structured dimension.

#### 4.4 Source score aggregation

File: `src/scoring/source_score.py`

Class: `SourceScorer`

Key functions:

- `score_dimension(dimension_name, embedding_score, bm25_score, structured_score)`
- `score_all_dimensions(embedding_scores, bm25_scores, structured_scores)`

Behavior:

- Combines the three retrieval sources into a single `SourceScore`.
- Each dimension has source-specific weights.
- The result includes the individual source values plus an aggregate `value`.

### 5. Stage 1 shortlist selection

File: `src/ranker.py`

Function:

- `Ranker.rank_candidates(top_k=1000)`

Behavior:

- Runs Stage 1 for all candidates.
- Sorts by `stage1_final_score`.
- Keeps only the top `top_k` candidates for Stage 2.

Current behavior:

- `top_k` is fixed inside the program.
- It is not user-controlled.
- Final output count is fixed at 100.

### 6. Cross-encoder reranking

File: `src/retrieval/cross_encoder_matcher.py`

Class: `CrossEncoderMatcher`

Key functions:

- `score_all_dimensions(dimensions, candidate)`

Behavior:

- Builds query-candidate pairs for the cross-encoder.
- Scores only the shortlisted candidates from Stage 1.
- Does not run over the full 100k pool.

Important detail:

- Cross-encoder scoring is bounded by the Stage 1 shortlist.
- This is the main reason the pipeline remains tractable.

### 7. Confidence scoring

File: `src/scoring/confidence.py`

Class: `ConfidenceScorer`

Key functions:

- `score_dimension(source_score)`
- `score_all_dimensions(source_scores)`

Behavior:

- Measures how consistent the three source signals are.
- Higher agreement between embedding, BM25, and structured scores increases confidence.
- Confidence is later used in final dimension scoring.

### 8. Final dimension scoring

File: `src/scoring/dimension_score.py`

Class: `DimensionScorer`

Key functions:

- `score_dimension(dimension_name, source_score, confidence_score, cross_encoder_score=0.0)`
- `score_all_dimensions(source_scores, confidence_scores, cross_encoder_scores)`

Behavior:

- Combines source score, confidence, and cross-encoder score.
- Applies dimension-specific cross-encoder weights.
- Returns final per-dimension scores.

### 9. Final candidate score

File: `src/scoring/candidate_score.py`

Dataclass:

- `CandidateScore`

Fields:

- `candidate_id`
- `dimension_scores`
- `final_score`

Behavior:

- Stores the final reranked score for each candidate.
- This is the object returned by the pipeline.

## End-to-End Execution Path

### CLI path

1. `rank.py:main()` parses arguments.
2. `rank.py:resolve_defaults()` chooses the candidate file path.
3. `Ranker.__init__()` loads the JD and candidates.
4. `Ranker.rank_candidates()` runs Stage 1 on the full candidate set.
5. Stage 1 uses:
   - `EmbeddingMatcher`
   - `BM25Matcher`
   - `StructuredMatcher`
   - `SourceScorer`
6. The shortlist is reranked with:
   - `CrossEncoderMatcher`
   - `ConfidenceScorer`
   - `DimensionScorer`
7. The top 100 results are printed and optionally written to JSON.

### Streamlit path

1. The user uploads a JD and a candidate file.
2. `app.py` parses the uploaded records.
3. `Ranker` is constructed with the uploaded data.
4. `Ranker.rank_candidates()` runs the same backend pipeline.
5. The app displays the final top 100 and offers a CSV download.

## Performance Characteristics

### Pool mode

- Candidate embeddings are precomputed in `artifacts/`.
- Candidate embedding lookup is O(1) per candidate.
- BM25 is computed once for the ranking run.
- Cross-encoder only runs on the shortlist.

### Sample mode

- Candidate embeddings are generated live from the uploaded sample.
- The sample size is capped at 100 internally.
- This mode is intended for quick ad hoc runs, not the 100k pool.

## Where Each Responsibility Lives

- JD parsing: `src/parser/jd_parser.py` -> `parse_jd()`
- Candidate parsing: `src/parser/candidate_parser.py` -> `parse_candidate()`, `load_candidates()`
- Dimension building: `src/dimensions/*.py`
- Stage 1 ranking: `src/ranker.py` -> `score_candidate_stage1()`
- Full ranking orchestration: `src/ranker.py` -> `rank_candidates()`
- Embedding retrieval: `src/retrieval/embedding_matcher.py`
- BM25 retrieval: `src/retrieval/bm25_matcher.py`
- Structured heuristics: `src/retrieval/structured_matcher.py`
- Cross-encoder reranking: `src/retrieval/cross_encoder_matcher.py`
- Source aggregation: `src/scoring/source_score.py`
- Confidence scoring: `src/scoring/confidence.py`
- Dimension-level reranking: `src/scoring/dimension_score.py`
- Final result object: `src/scoring/candidate_score.py`
- CLI entrypoint: `rank.py`
- Streamlit UI: `app.py`
- Precompute embeddings: `scripts/build_embeddings.py`

## Notes

- The ranking depth is fixed by program constants, not by user input.
- The user-facing choice is only the candidate source mode: sample upload or pool upload.
- The pool flow assumes the precomputed embedding artifacts are already available in `artifacts/`.
