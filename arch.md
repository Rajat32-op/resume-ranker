# Architecture

## Overview

The proposed solution is a hybrid, two-stage candidate ranking system designed to understand the intent behind a Job Description (JD) rather than relying on simple keyword matching. Instead of producing a single similarity score, the system evaluates every candidate across multiple hiring dimensions, combines evidence from different retrieval methods, and finally performs neural reranking on the strongest candidates.

The overall pipeline consists of:

1. JD Understanding
2. Candidate Processing
3. Hybrid Retrieval & Scoring
4. Cross Encoder Reranking
5. Final Ranking & Explainability

---

# JD Understanding

Rather than comparing the entire JD against the entire resume, the JD is first decomposed into several hiring dimensions. Each dimension represents an important hiring criterion.

Examples include:

* Technical Skills
* Ranking & Retrieval
* Production Experience
* Evaluation Frameworks
* Product Mindset
* Leadership
* Domain Experience
* Behavioral Signals
* Location

Each dimension contains a set of **semantic anchors**.

Anchors are representative concepts that define what the recruiter actually expects instead of merely matching exact keywords.

For example, the **Ranking & Retrieval** dimension contains anchors such as:

* embeddings
* retrieval
* recommendation systems
* vector databases
* FAISS
* Pinecone
* Qdrant
* Milvus

These anchors allow the system to understand semantic similarity even when the candidate uses different wording.

After mapping the JD to these dimensions, dynamic weights are generated automatically based on how strongly the JD emphasizes each requirement.

---

# Candidate Processing

Every candidate profile is parsed into a structured representation.

The parser extracts:

* Profile information
* Career history
* Education
* Skills
* Certifications
* Platform behavioral signals

A normalized textual representation (`raw_text`) is also created by combining all relevant information from the profile. This unified text is used by the semantic retrieval models.

---

# Stage 1: Hybrid Retrieval

Candidate evaluation is performed using three complementary retrieval methods.

## Semantic Embedding Matching

Semantic embeddings capture contextual similarity between a candidate profile and each JD dimension.

The system uses **BAAI/bge-small-en-v1.5** to encode both the JD dimensions and candidate profiles into dense vectors.

Similarity is computed using cosine similarity.

Unlike keyword search, semantic embeddings recognize conceptually similar experience even when different terminology is used.

Example:

A candidate describing

> "Built recommendation engines"

will still match a JD mentioning

> "Ranking and Retrieval"

even without sharing identical keywords.

---

## BM25 Matching

BM25 provides lexical matching based on exact words appearing in candidate profiles.

Instead of comparing against the entire JD, BM25 operates independently for each hiring dimension.

Each dimension is converted into a query consisting of:

* important anchors
* important JD phrases

A BM25 index is built over the candidate corpus once, allowing efficient retrieval of candidates containing highly relevant technical terms.

This complements semantic retrieval by rewarding precise matches for technologies such as:

* FAISS
* Milvus
* Pinecone
* LoRA
* NDCG
* BM25

---

## Structured Matching

Some hiring requirements cannot be measured reliably using text similarity alone.

The Structured Matcher evaluates explicit profile evidence using deterministic rules.

Examples include:

* Years of experience
* Production deployment experience
* Product vs service company background
* Evaluation framework experience
* Ranking and retrieval technologies

Each rule contributes evidence to one or more hiring dimensions.

Because structured matching relies on verified profile information rather than free text alone, it improves robustness against keyword stuffing and incomplete profiles.

---

# Score Fusion

Each hiring dimension receives three independent scores:

* Embedding Score
* BM25 Score
* Structured Score

These scores are combined using dimension-specific fusion weights.

Different dimensions rely on different retrieval methods.

For example:

* Technical Skills rely heavily on semantic embeddings.
* Production Experience relies more on structured evidence.
* Ranking & Retrieval benefits from both embeddings and BM25.

This produces a single Stage 1 score for every hiring dimension.

---

# Stage 2: Cross Encoder Reranking

Running a Cross Encoder over the complete candidate pool would be computationally expensive.

Instead, only the top candidates from Stage 1 are passed to the reranker.

The system uses **BAAI/bge-reranker-base** to jointly process the JD dimension and candidate profile, producing a deeper semantic relevance score.

Unlike embedding similarity, the Cross Encoder directly models interactions between the query and candidate, leading to significantly better ranking precision.

Its output is combined with the Stage 1 score using dimension-specific weights.

---

# Final Ranking

The final candidate score is computed as a weighted aggregation of all hiring dimensions.

Dimension weights are derived automatically from the Job Description, allowing the ranking system to adapt to different hiring priorities without changing the retrieval pipeline.

The candidates are then sorted by this final score to produce the ranked output.

---

# Explainability

Every recommendation includes a concise explanation generated directly from the computed scores.

The explanation highlights the candidate's strongest hiring dimensions along with important profile signals such as experience or recruiter responsiveness.

Since explanations are generated from computed evidence rather than a language model, every statement is traceable to the candidate profile, eliminating unsupported or hallucinated justifications.

---

# Scalability

The architecture is designed for efficient large-scale ranking.

For large candidate pools:

* Candidate embeddings are precomputed once.
* BM25 indexes are built once per corpus.
* Cross Encoder inference is limited to the top-ranked candidates.

For small uploaded candidate sets:

* Embeddings are generated on demand.
* The same ranking pipeline is reused without requiring precomputed artifacts.

This design enables both large-scale offline ranking and lightweight interactive deployment while remaining within the challenge's runtime constraints.
