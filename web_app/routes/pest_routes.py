from flask import Blueprint, render_template, request
from utils.pest_rules import calculate_pest_risk

pest_routes = Blueprint("pest_routes", __name__)

# POST route for pest prediction
@pest_routes.route("/predict_pest", methods=["POST"])
def predict_pest():
    pest_risk = None
    pest_score = None
    inputs = {}

    try:
        pest_inputs = {
            "stage": request.form.get("Stage"),
            "infestation": float(request.form.get("Infestation")),
            "variety": request.form.get("Variety"),
            "fertilizer": request.form.get("Fertilizer"),
            "pesticide": request.form.get("Pesticide"),
            "neighbor": request.form.get("Neighbor"),
            "weather": request.form.get("Weather"),
            "traps": request.form.get("Traps"),
            "susceptible_stages": ["Flowering", "Vegetative"]  # example rules
        }

        pest_risk, pest_score = calculate_pest_risk(pest_inputs)
        inputs = pest_inputs

    except Exception as e:
        pest_risk = f"Error: {str(e)}"

    return render_template(
        "profit_pest_estimation.html",
        pest_risk=pest_risk,
        pest_score=pest_score,
        inputs=inputs
    )
