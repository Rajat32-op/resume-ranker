from dataclasses import dataclass, field


@dataclass
class Dimension:
    """
    Represents one scoring dimension extracted from the JD.
    Contains only semantic information.
    """

    name: str
    weight: float

    # Keywords / semantic anchors associated with this dimension
    anchors: list[str] = field(default_factory=list)

    # Paragraphs from the JD mapped to this dimension
    jd_paragraphs: list[str] = field(default_factory=list)

    # Optional description for debugging / explainability
    description: str = ""

    anchor_hits: int = 0