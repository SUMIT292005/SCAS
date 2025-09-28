import json, os

def recommend_treatment(crop, disease):
    rules_path = os.path.join("models", "model7_treatment_advisor_rules.json")
    with open(rules_path, "r") as f:
        rules = json.load(f)
    if crop in rules and disease in rules[crop]:
        return rules[crop][disease]
    return {"Message": "No treatment information available."}
