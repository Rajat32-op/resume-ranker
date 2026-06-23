from src.ranker import Ranker

ranker = Ranker(
    "data/job_description.docx",
    "data/sample_candidates.json"
)

results = ranker.rank_candidates()

for result in results[:20]:

    print(
        result.candidate_id,
        round(result.final_score, 3)
    )