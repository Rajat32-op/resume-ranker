from src.parser.jd_parser import parse_jd


jd = parse_jd("data/job_description.docx")

print("Title:", jd.title)
print("Company:", jd.company)
print("Location:", jd.location)
print("Employment Type:", jd.employment_type)
print("Experience:", jd.experience_min_years, "-", jd.experience_max_years)
print()

print("First 5 paragraphs:")
for req in jd.requirements[:5]:
    print(req.text)
    print()