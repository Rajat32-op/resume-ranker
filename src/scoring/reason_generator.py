from src.parser.schema import Candidate
from src.scoring.dimension_score import DimensionScore


DIMENSION_NAMES = {
    "TECHNICAL_SKILLS": "technical skills",
    "RANKING_RETRIEVAL": "ranking/retrieval",
    "PRODUCTION_EXPERIENCE": "production systems",
    "EVALUATION_FRAMEWORKS": "evaluation frameworks",
    "PRODUCT_MINDSET": "product mindset",
    "DOMAIN_EXPERIENCE": "domain experience",
    "LEADERSHIP": "leadership"
}


class ReasonGenerator:

    def generate(
            self,
            candidate: Candidate,
            dimension_scores: dict[str, DimensionScore]
    ) -> str:

        top_dimensions = sorted(
            dimension_scores.items(),
            key=lambda x: x[1].final_score,
            reverse=True
        )

        strengths = []

        for name, _ in top_dimensions:

            if name in DIMENSION_NAMES:

                strengths.append(
                    DIMENSION_NAMES[name]
                )

            if len(strengths) == 2:
                break

        reason = (
            f"{candidate.profile.current_title} with "
            f"{candidate.profile.years_of_experience:.1f} years experience. "
        )

        if strengths:

            reason += (
                "Strong "
                + " and ".join(strengths)
                + ". "
            )

        if candidate.redrob_signals.open_to_work_flag:

            reason += "Open to work. "

        if (
            candidate.redrob_signals.recruiter_response_rate
            >= 0
        ):

            reason += (
                f"Recruiter response rate "
                f"{candidate.redrob_signals.recruiter_response_rate:.2f}."
            )

        return reason.strip()