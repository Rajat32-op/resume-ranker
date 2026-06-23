from dataclasses import dataclass, field

from src.parser.schema import Candidate


@dataclass
class MatchResult:
    score: float
    evidence: list[str] = field(default_factory=list)

RANKING_TERMS = {
    "ranking",
    "retrieval",
    "search",
    "recommendation",
    "recommendation system",
    "embeddings",
    "vector database",
    "faiss",
    "pinecone",
    "qdrant",
    "milvus",
    "weaviate",
    "elasticsearch",
    "opensearch",
    "bm25"
}


EVALUATION_TERMS = {
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "ab testing",
    "offline benchmark",
    "evaluation framework"
}


SERVICE_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "tech mahindra",
    "hcl"
}

def score_ranking_retrieval(
        candidate: Candidate
) -> MatchResult:

    text = candidate.raw_text.lower()

    score = 0
    evidence = []

    for term in RANKING_TERMS:

        if term in text:

            score += 0.1
            evidence.append(term)

    score = min(score, 1.0)

    return MatchResult(
        score=score,
        evidence=evidence
    )

def score_production_experience(
        candidate: Candidate
) -> MatchResult:

    text = candidate.raw_text.lower()

    score = 0
    evidence = []

    if candidate.profile.years_of_experience >= 5:
        score += 0.2
        evidence.append("5+ years experience")

    for phrase in [
        "production",
        "deployed",
        "real users",
        "scale",
        "pipeline"
    ]:

        if phrase in text:
            score += 0.15
            evidence.append(phrase)

    score = min(score, 1.0)

    return MatchResult(
        score=score,
        evidence=evidence
    )

def score_evaluation_frameworks(
        candidate: Candidate
) -> MatchResult:

    text = candidate.raw_text.lower()

    score = 0
    evidence = []

    for term in EVALUATION_TERMS:

        if term in text:
            score += 0.25
            evidence.append(term)

    score = min(score, 1.0)

    return MatchResult(
        score=score,
        evidence=evidence
    )

def score_product_mindset(
        candidate: Candidate
) -> MatchResult:

    score = 0
    evidence = []

    company_names = [
        x.company.lower()
        for x in candidate.career_history
    ]

    if all(
        company not in SERVICE_COMPANIES
        for company in company_names
    ):
        score += 0.5
        evidence.append("product company experience")

    if candidate.profile.years_of_experience >= 5:
        score += 0.2
        evidence.append("senior engineer")

    return MatchResult(
        score=min(score, 1.0),
        evidence=evidence
    )

def score_all_dimensions(
        candidate: Candidate
):

    return {

        "RANKING_RETRIEVAL":
            score_ranking_retrieval(candidate),

        "PRODUCTION_EXPERIENCE":
            score_production_experience(candidate),

        "EVALUATION_FRAMEWORKS":
            score_evaluation_frameworks(candidate),

        "PRODUCT_MINDSET":
            score_product_mindset(candidate)

    }

