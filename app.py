import json
import tempfile

import pandas as pd
import streamlit as st

from src.parser.candidate_parser import load_candidates_from_records
from src.ranker import Ranker

DEFAULT_EMBEDDINGS_PATH = "artifacts/candidate_embeddings.npy"
DEFAULT_IDS_PATH = "artifacts/candidate_ids.npy"
STAGE1_TOP_K = 1000
FINAL_OUTPUT_TOP_N = 100


st.set_page_config(
    page_title="Resume Filter",
    page_icon=None,
    layout="wide"
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(194, 65, 12, 0.12), transparent 32%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.12), transparent 28%),
                linear-gradient(180deg, #fffaf3 0%, #fbf6ee 45%, #f4ece1 100%);
            color: #1f2937;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1f2937 45%, #7c2d12 120%);
            color: white;
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin-top: 0.6rem;
            max-width: 54rem;
            color: rgba(255, 255, 255, 0.84);
            font-size: 1rem;
            line-height: 1.6;
        }
        .card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
        }
        .stRadio [role="radiogroup"] {
            gap: 0.5rem;
        }
        .stRadio label {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(15, 23, 42, 0.1);
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
        }
        .stButton button {
            background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%);
            color: white;
            border: none;
            border-radius: 999px;
            box-shadow: 0 12px 30px rgba(194, 65, 12, 0.28);
        }
        .stButton button:hover {
            background: linear-gradient(135deg, #9a3412 0%, #c2410c 100%);
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Resume Filter</h1>
        <p>
            Upload a JD and candidate set, then run either the live embedding path for a small sample
            or the precomputed embedding path for the full pool.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Inputs")

    candidate_mode = st.radio(
        "Candidate mode",
        options=["sample", "pool"],
        format_func=lambda value: "Sample upload (live embeddings)" if value == "sample" else "100k pool (precomputed embeddings)",
        horizontal=True,
    )

    jd_file = st.file_uploader(
        "Job description (.docx)",
        type=["docx"],
        accept_multiple_files=False,
    )

    if candidate_mode == "sample":
        candidate_file = st.file_uploader(
            "Candidates JSON (.json)",
            type=["json"],
            accept_multiple_files=False,
            help="Upload a JSON array of up to 100 candidates."
        )
    else:
        candidate_file = st.file_uploader(
            "Candidates JSONL (.jsonl)",
            type=["jsonl"],
            accept_multiple_files=False,
            help="Upload the large candidate pool file. Candidate ids must match the precomputed embeddings."
        )

    st.caption("The ranking depth is fixed in code. The app always returns the top 100 results.")

    run_button = st.button("Run ranking", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Output")

    if run_button:
        if jd_file is None or candidate_file is None:
            st.error("Upload both the JD and the candidate file before running the pipeline.")
        else:
            candidate_bytes = candidate_file.getvalue()
            if candidate_mode == "sample":
                candidate_records = json.loads(candidate_bytes.decode("utf-8"))
                if isinstance(candidate_records, dict):
                    candidate_records = [candidate_records]
                candidate_records = candidate_records[:100]
                candidate_objects = load_candidates_from_records(candidate_records)
            else:
                candidate_text = candidate_bytes.decode("utf-8")
                candidate_records = [
                    json.loads(line)
                    for line in candidate_text.splitlines()
                    if line.strip()
                ]
                candidate_objects = load_candidates_from_records(candidate_records)

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as jd_temp:
                jd_temp.write(jd_file.getvalue())
                jd_path = jd_temp.name

            with st.spinner("Running ranking pipeline..."):
                ranker = Ranker(
                    jd_path=jd_path,
                    candidates=candidate_objects,
                    use_precomputed_embeddings=candidate_mode == "pool",
                    embeddings_path=DEFAULT_EMBEDDINGS_PATH,
                    ids_path=DEFAULT_IDS_PATH,
                )
                results = ranker.rank_candidates(top_k=STAGE1_TOP_K)

            rows = []
            for result in results[: FINAL_OUTPUT_TOP_N]:
                rows.append(
                    {
                        "candidate_id": result.candidate_id,
                        "final_score": result.final_score,
                    }
                )

            if rows:
                frame = pd.DataFrame(rows)
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name="ranking_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No results returned.")
    else:
        st.write("Upload files and run the ranking pipeline to see results here.")

    st.markdown("</div>", unsafe_allow_html=True)
