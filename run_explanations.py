from explanation_layer import explain_scored_record
import json
import csv

values_by_id = {}
with open("Synthetic Patients Processed.csv") as f:
    for row in csv.DictReader(f):
        values_by_id[row["patient_id"]] = row

with open("patients_scored.json") as f:
    patients = json.load(f)

for p in patients:
    extra = values_by_id.get(p["patient_id"], {})
    for key, val in extra.items():
        p.setdefault(key, val)

for p in patients[:10]:
    result = explain_scored_record(p, use_llm=True)
    print(p["patient_id"], "->", result["explanation"])
    print("   suggestion:", result["suggestion"])
    for point in result["talking_points"]:
        print("   -", point)
    print()
