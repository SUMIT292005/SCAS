def calculate_pest_risk(inputs):
    score = 0

    # Crop + stage
    if inputs["stage"] in inputs.get("susceptible_stages", []):
        score += 2
    else:
        score += 1

    # Infestation %
    inf = inputs["infestation"]
    if inf < 10: score += 1
    elif inf < 30: score += 2
    else: score += 3

    # Variety
    score += {"Resistant": 0, "Moderately Resistant": 1, "Susceptible": 2}.get(inputs["variety"], 0)

    # Fertilizer
    if inputs["fertilizer"] == "Heavy N": score += 2

    # Pesticide
    if inputs["pesticide"] == "Recent Spray": score -= 1

    # Neighbor alerts
    score += {"None": 0, "Nearby": 1, "Same Block": 2}.get(inputs["neighbor"], 0)

    # Weather
    score += {"Unfavorable": 0, "Neutral": 1, "Favorable": 2}.get(inputs["weather"], 0)

    # Trap counts
    traps = inputs["traps"]
    if traps == "Low": score += 1
    elif traps == "Medium": score += 2
    elif traps == "High": score += 3

    # Final category
    if score <= 4:
        return "Low", score
    elif score <= 8:
        return "Medium", score
    else:
        return "High", score
