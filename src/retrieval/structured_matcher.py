from dataclasses import dataclass, field

from src.parser.schema import Candidate


@dataclass
class MatchResult:
    score: float
    evidence: list[str] = field(default_factory=list)


class StructuredMatcher:

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
            self,
            candidate: Candidate
    ) -> MatchResult:

        text = candidate.lower_text or candidate.raw_text.lower()

        score = 0
        evidence = []

        for term in self.RANKING_TERMS:

            if term in text:
                score += 0.1
                evidence.append(term)

        return MatchResult(
            score=min(score, 1.0),
            evidence=evidence
        )

    def score_production_experience(
            self,
            candidate: Candidate
    ) -> MatchResult:

        text = candidate.lower_text or candidate.raw_text.lower()

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

        return MatchResult(
            score=min(score, 1.0),
            evidence=evidence
        )

    def score_evaluation_frameworks(
            self,
            candidate: Candidate
    ) -> MatchResult:

        text = candidate.lower_text or candidate.raw_text.lower()

        score = 0
        evidence = []

        for term in self.EVALUATION_TERMS:

            if term in text:
                score += 0.25
                evidence.append(term)

        return MatchResult(
            score=min(score, 1.0),
            evidence=evidence
        )

    def score_product_mindset(
            self,
            candidate: Candidate
    ) -> MatchResult:

        score = 0
        evidence = []

        company_names = [
            x.company.lower()
            for x in candidate.career_history
        ]

        if all(
                company not in self.SERVICE_COMPANIES
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
            self,
            candidate: Candidate
    ) -> dict[str, MatchResult]:

        return {

            "RANKING_RETRIEVAL":
                self.score_ranking_retrieval(candidate),

            "PRODUCTION_EXPERIENCE":
                self.score_production_experience(candidate),

            "EVALUATION_FRAMEWORKS":
                self.score_evaluation_frameworks(candidate),

            "PRODUCT_MINDSET":
                self.score_product_mindset(candidate)

        }

    def score_all_dimensions_values(
            self,
            candidate: Candidate
    ) -> dict[str, float]:
        text = candidate.lower_text or candidate.raw_text.lower()

        ranking_score = 0.0
        for term in self.RANKING_TERMS:
            if term in text:
                ranking_score += 0.1

        production_score = 0.0
        if candidate.profile.years_of_experience >= 5:
            production_score += 0.2
        for phrase in ["production", "deployed", "real users", "scale", "pipeline"]:
            if phrase in text:
                production_score += 0.15

        evaluation_score = 0.0
        for term in self.EVALUATION_TERMS:
            if term in text:
                evaluation_score += 0.25

        company_names = [x.company.lower() for x in candidate.career_history]
        product_score = 0.0
        if all(company not in self.SERVICE_COMPANIES for company in company_names):
            product_score += 0.5
        if candidate.profile.years_of_experience >= 5:
            product_score += 0.2

        return {
            "RANKING_RETRIEVAL": min(ranking_score, 1.0),
            "PRODUCTION_EXPERIENCE": min(production_score, 1.0),
            "EVALUATION_FRAMEWORKS": min(evaluation_score, 1.0),
            "PRODUCT_MINDSET": min(product_score, 1.0)
        }