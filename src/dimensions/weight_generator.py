from .dimension_schema import Dimension


EMPHASIS_WORDS = [
    "must",
    "required",
    "absolutely need",
    "need",
    "explicitly",
    "disqualifier",
    "will not move forward",
    "do not want"
]


def generate_weights(
        dimensions: list[Dimension]
) -> list[Dimension]:

    NON_WEIGHTED_DIMENSIONS = {
        "LOCATION",
        "BEHAVIORAL_SIGNALS"
    }

    scores = []

    for dimension in dimensions:

        if dimension.name in NON_WEIGHTED_DIMENSIONS:
            scores.append(0)
            continue

        paragraph_score = len(dimension.jd_paragraphs)

        anchor_score = dimension.anchor_hits

        emphasis_score = 0

        for paragraph in dimension.jd_paragraphs:

            p = paragraph.lower()

            for word in EMPHASIS_WORDS:
                if word in p:
                    emphasis_score += 3

        total_score = (
            paragraph_score
            + anchor_score
            + emphasis_score
        )

        scores.append(total_score)

    total = sum(scores)

    if total == 0:
        return dimensions

    for dimension, score in zip(dimensions, scores):
        dimension.weight = score / total

    return dimensions