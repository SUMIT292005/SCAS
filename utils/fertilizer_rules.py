import json

# Load JSON once
with open("models/model4_fertilizer_advisor_rules.json") as f:
    FERTILIZER_RULES = json.load(f)


def get_growth_fertilizer(crop: str, N: float, P: float, K: float, pH: float,
                          rainfall: float = None, temperature: float = None, humidity: float = None):
    """
    Returns fertilizer recommendations + weather warnings
    Inputs:
        crop, N, P, K, pH (mandatory)
        rainfall (mm/day), temperature (°C), humidity (%) [optional for warnings]
    """

    # ---------- Fertilizer Rules ----------
    if crop not in FERTILIZER_RULES:
        return {"fertilizers": ["No rules found for this crop"], "warnings": []}

    crop_rules = FERTILIZER_RULES[crop]
    suggestions = []

    # Nitrogen
    if N < 50:
        suggestions.append(crop_rules["N"]["low"])
    elif N < 100:
        suggestions.append(crop_rules["N"]["medium"])
    else:
        suggestions.append(crop_rules["N"]["high"])

    # Phosphorus
    if P < 30:
        suggestions.append(crop_rules["P"]["low"])
    elif P < 60:
        suggestions.append(crop_rules["P"]["medium"])
    else:
        suggestions.append(crop_rules["P"]["high"])

    # Potassium
    if K < 40:
        suggestions.append(crop_rules["K"]["low"])
    elif K < 80:
        suggestions.append(crop_rules["K"]["medium"])
    else:
        suggestions.append(crop_rules["K"]["high"])

    # pH
    if pH < 6.0:
        suggestions.append(crop_rules["pH"]["acidic"])
    elif pH <= 7.5:
        suggestions.append(crop_rules["pH"]["neutral"])
    else:
        suggestions.append(crop_rules["pH"]["alkaline"])

    # ---------- Weather Warnings ----------
    warnings = []
    if "_meta" in FERTILIZER_RULES and "weather_rules" in FERTILIZER_RULES["_meta"]:
        wr = FERTILIZER_RULES["_meta"]["weather_rules"]["delay_if"]

        if rainfall is not None and rainfall >= 20:
            warnings.append("🌧 Heavy rain expected – delay fertilizer application to avoid nutrient loss.")
        if temperature is not None and (temperature >= 40 or temperature <= 10):
            warnings.append("🔥 Extreme temperature – delay fertilizer application. Prefer moderate (15–35 °C).")
        if humidity is not None and humidity >= 90:
            warnings.append("💧 Very high humidity – avoid foliar sprays, may promote fungal diseases.")

    return {
        "fertilizers": suggestions,
        "warnings": warnings
    }

