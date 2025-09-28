from flask import Blueprint, render_template, request
from web_app.dashboard_helpers import recommend_treatment

treatment_routes = Blueprint("treatment_routes", __name__)

@treatment_routes.route("/treatment_advisor", methods=["GET"])
def show_treatment_advisor():
    # Just load the page with no recommendation initially
    return render_template("treatment_advisor.html", recommendation=None)

@treatment_routes.route("/recommend_treatment", methods=["POST"])
def recommend_treatment_route():
    recommendation = None
    crop = request.form.get("crop")
    disease = request.form.get("disease")
    if crop and disease:
        recommendation = recommend_treatment(crop, disease)

    return render_template("treatment_advisor.html", recommendation=recommendation)
