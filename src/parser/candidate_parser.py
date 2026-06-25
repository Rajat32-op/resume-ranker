import json
from pathlib import Path

from typing import Iterable

from .schema import (
    Candidate,
    Profile,
    CareerEntry,
    EducationEntry,
    Skill,
    Certification,
    Language,
    SalaryRange,
    RedrobSignals,
)


def parse_candidate(candidate_json: dict) -> Candidate:

    # ---------- profile ----------
    p = candidate_json["profile"]

    profile = Profile(
        anonymized_name=p["anonymized_name"],
        headline=p["headline"],
        summary=p["summary"],
        location=p["location"],
        country=p["country"],
        years_of_experience=p["years_of_experience"],
        current_title=p["current_title"],
        current_company=p["current_company"],
        current_company_size=p["current_company_size"],
        current_industry=p["current_industry"]
    )

    # ---------- career ----------
    career_history = []

    for c in candidate_json["career_history"]:
        career_history.append(
            CareerEntry(
                company=c["company"],
                title=c["title"],
                start_date=c["start_date"],
                end_date=c["end_date"],
                duration_months=c["duration_months"],
                is_current=c["is_current"],
                industry=c["industry"],
                company_size=c["company_size"],
                description=c["description"]
            )
        )

    # ---------- education ----------
    education = []

    for e in candidate_json["education"]:
        education.append(
            EducationEntry(
                institution=e["institution"],
                degree=e["degree"],
                field_of_study=e["field_of_study"],
                start_year=e["start_year"],
                end_year=e["end_year"],
                grade=e.get("grade"),
                tier=e.get("tier", "unknown")
            )
        )

    # ---------- skills ----------
    skills = []

    for s in candidate_json["skills"]:
        skills.append(
            Skill(
                name=s["name"],
                proficiency=s["proficiency"],
                endorsements=s["endorsements"],
                duration_months=s.get("duration_months", 0)
            )
        )

    # ---------- certifications ----------
    certifications = []

    for c in candidate_json.get("certifications", []):
        certifications.append(
            Certification(
                name=c["name"],
                issuer=c["issuer"],
                year=c["year"]
            )
        )

    # ---------- languages ----------
    languages = []

    for l in candidate_json.get("languages", []):
        languages.append(
            Language(
                language=l["language"],
                proficiency=l["proficiency"]
            )
        )

    # ---------- signals ----------
    r = candidate_json["redrob_signals"]

    redrob_signals = RedrobSignals(
        profile_completeness_score=r["profile_completeness_score"],
        signup_date=r["signup_date"],
        last_active_date=r["last_active_date"],
        open_to_work_flag=r["open_to_work_flag"],
        profile_views_received_30d=r["profile_views_received_30d"],
        applications_submitted_30d=r["applications_submitted_30d"],
        recruiter_response_rate=r["recruiter_response_rate"],
        avg_response_time_hours=r["avg_response_time_hours"],
        skill_assessment_scores=r["skill_assessment_scores"],
        connection_count=r["connection_count"],
        endorsements_received=r["endorsements_received"],
        notice_period_days=r["notice_period_days"],
        expected_salary_range_inr_lpa=SalaryRange(
            min=r["expected_salary_range_inr_lpa"]["min"],
            max=r["expected_salary_range_inr_lpa"]["max"]
        ),
        preferred_work_mode=r["preferred_work_mode"],
        willing_to_relocate=r["willing_to_relocate"],
        github_activity_score=r["github_activity_score"],
        search_appearance_30d=r["search_appearance_30d"],
        saved_by_recruiters_30d=r["saved_by_recruiters_30d"],
        interview_completion_rate=r["interview_completion_rate"],
        offer_acceptance_rate=r["offer_acceptance_rate"],
        verified_email=r["verified_email"],
        verified_phone=r["verified_phone"],
        linkedin_connected=r["linkedin_connected"]
    )

    # ---------- raw text ----------
    raw_text_parts = [
        profile.headline,
        profile.summary,
        profile.current_title
    ]

    raw_text_parts.extend([s.name for s in skills])

    for c in career_history:
        raw_text_parts.append(c.title)
        raw_text_parts.append(c.description)

    for e in education:
        raw_text_parts.append(e.degree)
        raw_text_parts.append(e.field_of_study)

    raw_text = "\n".join(raw_text_parts)

    return Candidate(
        candidate_id=candidate_json["candidate_id"],
        profile=profile,
        career_history=career_history,
        education=education,
        skills=skills,
        certifications=certifications,
        languages=languages,
        redrob_signals=redrob_signals,
        raw_text=raw_text
    )


def load_candidates_from_records(
        candidate_records: Iterable[dict]
):
    return [parse_candidate(candidate_json) for candidate_json in candidate_records]


def load_candidates(
        path: str,
        file_format: str | None = None,
        limit: int | None = None
):
    path_obj = Path(path)

    if file_format is None:
        file_format = "jsonl" if path_obj.suffix.lower() == ".jsonl" else "json"

    candidate_records = []

    with open(path_obj, encoding="utf-8") as f:

        if file_format == "jsonl":

            for line in f:

                line = line.strip()

                if not line:
                    continue

                candidate_records.append(json.loads(line))

                if limit is not None and len(candidate_records) >= limit:
                    break

        else:

            candidates_json = json.load(f)

            if isinstance(candidates_json, dict):
                candidates_json = [candidates_json]

            if limit is not None:
                candidates_json = candidates_json[:limit]

            candidate_records = list(candidates_json)

    return load_candidates_from_records(candidate_records)