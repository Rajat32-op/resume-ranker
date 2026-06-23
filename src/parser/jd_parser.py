from pathlib import Path
import re
from docx import Document # type: ignore

from .jd_schema import (
    JD,
    Requirement,
)


def parse_jd(docx_path: str | Path) -> JD:
    """
    Parse a JD docx into a JD object.
    """

    doc = Document(docx_path)

    paragraphs = [
        p.text.strip()
        for p in doc.paragraphs
        if p.text.strip()
    ]

    raw_text = "\n".join(paragraphs)

    # ---------------- title ----------------
    title = ""
    if paragraphs:
        first = paragraphs[0]

        m = re.search(
            r"Job Description:\s*(.+)",
            first,
            re.IGNORECASE
        )

        if m:
            title = m.group(1).strip()
        else:
            title = first

    # ---------------- company ----------------
    company = ""

    m = re.search(
        r"Company:\s*(.+)",
        raw_text,
        re.IGNORECASE
    )

    if m:
        company = m.group(1).strip()

    # ---------------- location ----------------
    location = ""

    m = re.search(
        r"Location:\s*(.+)",
        raw_text,
        re.IGNORECASE
    )

    if m:
        location = m.group(1).strip()

    # ---------------- employment type ----------------
    employment_type = ""

    m = re.search(
        r"Employment Type:\s*(.+)",
        raw_text,
        re.IGNORECASE
    )

    if m:
        employment_type = m.group(1).strip()

    # ---------------- experience years ----------------
    experience_min_years = None
    experience_max_years = None

    m = re.search(
        r"Experience Required:\s*(\d+)\s*[–-]\s*(\d+)",
        raw_text,
        re.IGNORECASE
    )

    if m:
        experience_min_years = float(m.group(1))
        experience_max_years = float(m.group(2))

    # ---------------- requirements ----------------
    requirements = [
        Requirement(text=p)
        for p in paragraphs
    ]

    jd = JD(
        title=title,
        company=company,
        location=location,
        employment_type=employment_type,
        experience_min_years=experience_min_years,
        experience_max_years=experience_max_years,
        requirements=requirements,
        raw_text=raw_text
    )

    return jd