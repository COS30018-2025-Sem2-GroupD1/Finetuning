import gzip, json

path = "data/healthcaremagic_distillation_softlabels.jsonl.gz"

# Count rows
count = 0
with gzip.open(path, "rt", encoding="utf-8") as f:
    for _ in f:
        count += 1
print("Total rows:", count)

# Show the first example
with gzip.open(path, "rt", encoding="utf-8") as f:
    first = json.loads(next(f))
print(json.dumps(first, indent=2)[:1000])  # pretty-print truncated
