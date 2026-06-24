import argparse
import pandas as pd
from src.ranker import Ranker

def generate_reasoning(candidate_score):
    """
    Generate a simple reasoning string based on top performing dimensions.
    """
    top_dims = sorted(
        candidate_score.dimension_scores.items(),
        key=lambda x: x[1].final_score,
        reverse=True
    )[:3]
    
    reasons = []
    for dim_name, scores in top_dims:
        reasons.append(f"{dim_name.lower().replace('_', ' ')} ({scores.final_score:.2f})")
    
    return f"Strong match in: {', '.join(reasons)}."

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for a JD.")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates JSON/JSONL file.")
    parser.add_argument("--jd", type=str, default="data/job_description.docx", help="Path to JD docx file.")
    parser.add_argument("--out", type=str, default="submission.csv", help="Output path for the CSV.")
    parser.add_argument("--top_k", type=int, default=1000, help="Number of candidates to rerank in Stage 2.")
    
    args = parser.parse_args()
    
    print(f"Loading Ranker with JD: {args.jd}")
    ranker = Ranker(args.jd, args.candidates)
    
    print(f"Ranking candidates (Stage 2 top_k={args.top_k})...")
    results = ranker.rank_candidates(top_k=args.top_k)
    
    output_data = []
    for i, res in enumerate(results):
        output_data.append({
            "candidate_id": res.candidate_id,
            "rank": i + 1,
            "score": round(res.final_score, 4),
            "reasoning": generate_reasoning(res)
        })
    
    df = pd.DataFrame(output_data)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} results to {args.out}")

if __name__ == "__main__":
    main()
