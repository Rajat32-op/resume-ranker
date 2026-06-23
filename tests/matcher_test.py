# tests/bm25_matcher_test.py

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

from src.retrieval.bm25_matcher import (
    BM25Matcher
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

    candidate_jsons = json.load(
        f
    )


candidates = [

    parse_candidate(
        candidate_json
    )

    for candidate_json in candidate_jsons
]


matcher = BM25Matcher(
    candidates
)

results = matcher.score_all_dimensions(
    dimensions
)


candidate_id = candidates[0].candidate_id

for dimension_name in results:

    print(
        f"{dimension_name:25}",
        f"{results[dimension_name][candidate_id]:.3f}"
    )