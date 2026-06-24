# tests/cross_encoder_test.py

import json

from src.parser.jd_parser import parse_jd
from src.parser.candidate_parser import parse_candidate

from src.dimensions.anchor_builder import (
    build_dimension_templates
)
from src.dimensions.dimension_mapper import (
    map_jd_to_dimensions
)
from src.dimensions.weight_generator import (
    generate_weights
)

from src.retrieval.cross_encoder_matcher import (
    CrossEncoderMatcher
)


jd = parse_jd(
    "data/job_description.docx"
)

dimensions = build_dimension_templates()

dimensions = map_jd_to_dimensions(
    jd,
    dimensions
)

dimensions = generate_weights(
    dimensions
)


with open(
        "data/sample_candidates.json"
) as f:

    candidate_json = json.load(
        f
    )[0]


candidate = parse_candidate(
    candidate_json
)


matcher = CrossEncoderMatcher()

scores = matcher.score_all_dimensions(
    dimensions,
    candidate
)


for name, score in sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
):

    print(
        f"{name:25} {score:.3f}"
    )