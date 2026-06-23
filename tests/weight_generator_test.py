from src.parser.jd_parser import parse_jd
from src.dimensions.anchor_builder import build_dimension_templates
from src.dimensions.dimension_mapper import map_jd_to_dimensions
from src.dimensions.weight_generator import generate_weights


jd = parse_jd("data/job_description.docx")

dimensions = build_dimension_templates()

dimensions = map_jd_to_dimensions(
    jd,
    dimensions
)

dimensions = generate_weights(
    dimensions
)


print()

for dim in sorted(
        dimensions,
        key=lambda x: x.weight,
        reverse=True):

    print(
        f"{dim.name:25}",
        f"weight={dim.weight:.3f}",
        f"paragraphs={len(dim.jd_paragraphs)}",
        f"anchor_hits={dim.anchor_hits}"
    )