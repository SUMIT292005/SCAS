from flask import Blueprint, render_template, request, jsonify
from web_app.dashboard_helpers import suggest_growth_fertilizer
from utils.api_weather import get_all_districts, get_weather

fertilizer_routes = Blueprint("fertilizer_routes", __name__)

@fertilizer_routes.route("/fertilizer_advisor", methods=["GET", "POST"])
def fertilizer_advisor():
    suggestions = None
    weather_data = None
    districts = get_all_districts()

    if request.method == "POST":
        crop = request.form.get("crop")
        N = request.form.get("N")
        P = request.form.get("P")
        K = request.form.get("K")
        pH = request.form.get("pH")

        rainfall = request.form.get("rainfall")
        temperature = request.form.get("temperature")
        humidity = request.form.get("humidity")

        # If automatic weather used, override with selected district
        if request.form.get("use_auto_weather") == "on":
            district = request.form.get("district")
            if district:
                weather_data = get_weather(district)
                rainfall = weather_data["rainfall"]
                temperature = weather_data["temperature"]
                humidity = weather_data["humidity"]

        suggestions = suggest_growth_fertilizer(
            crop, N, P, K, pH,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity
        )

    return render_template(
        "fertilizer_advisor.html",
        suggestions=suggestions,
        districts=districts,
        weather_data=weather_data
    )

# ------------------ API endpoint for dynamic weather fetch ------------------ #
@fertilizer_routes.route("/get_weather_data", methods=["POST"])
def get_weather_data():
    data = request.get_json()
    district = data.get("district")
    try:
        weather = get_weather(district)
        return jsonify({"success": True, "weather": weather})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
