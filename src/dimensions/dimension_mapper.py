import re

from src.parser.jd_schema import JD
from .dimension_schema import Dimension


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def map_jd_to_dimensions(
    jd: JD,
    dimensions: list[Dimension]
) -> list[Dimension]:
    """
    Assign JD paragraphs to dimensions using anchor matching.
    A paragraph may belong to multiple dimensions.
    """

    for paragraph_obj in jd.requirements:

        paragraph = paragraph_obj.text
        paragraph_norm = normalize(paragraph)

        for dimension in dimensions:

            num_hits = 0

            for anchor in dimension.anchors:

                if anchor.lower() in paragraph_norm:
                    num_hits += 1

            if num_hits > 0:
                dimension.jd_paragraphs.append(paragraph)
                dimension.anchor_hits += num_hits

    return dimensions