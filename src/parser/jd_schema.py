from dataclasses import dataclass, field


@dataclass
class Requirement:
    text: str


@dataclass
class PreferredQualification:
    text: str


@dataclass
class Responsibility:
    text: str


@dataclass
class JD:
    title: str = ""

    company: str = ""
    location: str = ""
    employment_type: str = ""

    experience_min_years: float | None = None
    experience_max_years: float | None = None

    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)

    requirements: list[Requirement] = field(default_factory=list)
    preferred_qualifications: list[PreferredQualification] = field(default_factory=list)
    responsibilities: list[Responsibility] = field(default_factory=list)

    raw_text: str = ""