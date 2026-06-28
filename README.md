# Resume Filter - Intelligent Candidate Ranking System

## Installation

Clone the repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Preprocessing (Required for Pool Mode)

Before ranking the full **100K candidate pool**, run the preprocessing script **once**.

This step downloads the required models and precomputed artifacts, and builds the BM25 index required for fast inference.

```bash
python preprocess.py
```

> **Note:** Preprocessing is a one-time setup step and may take several minutes. The actual ranking step is optimized to complete within the challenge runtime limit.

---

## Running the Ranker


---

### Pool Mode (100K Candidates)

Use this mode to rank the complete candidate pool.

```bash
python rank.py --candidate-mode=pool
```

The generated ranking file will be saved in the **`outputs/`** directory.

---

## Hosted Demo

To check a sample (<100):

[**Hugging Face Space:**](https://huggingface.co/spaces/Rajat3206/resume_analysis)
