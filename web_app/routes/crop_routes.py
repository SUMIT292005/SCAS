from flask import Blueprint, render_template, request, jsonify
from utils.api_weather import get_weather, get_all_districts
from utils.model_helpers import load_model1_crop_recommender
from web_app.dashboard_helpers import recommend_crops, get_average_soil, average_soil_df

bp = Blueprint("crop_routes", __name__)



@bp.route("/")
def index():
    # Load dataset metrics (you already have these in your dataset or helpers)
    total_crop_records = 6713  # Example, replace with actual dataset size
    unique_crops = 29        # Replace with dataset logic
    total_markets = 656
    total_districts = 283
    
    # Top crops data (replace with your actual top crops data)
    top_crops = [
        {'name': 'Sugarcane', 'count': 1010},
        {'name': 'Wheat', 'count': 859},
        {'name': 'Cotton', 'count': 750}
    ]
    
    # Environmental data ranges (based on your dataset)
    env_data = {
        'temperature': {'min': 8.8, 'max': 43.7},
        'humidity': {'min': 14.3, 'max': 100},
        'rainfall': {'min': 20.2, 'max': 1700},
        'pH': {'min': 3.5, 'max': 9.94}
    }
    
    # Other relevant info you want in the dashboard
    return render_template(
        "index.html",  # The dashboard template you will create
        total_crop_records=total_crop_records,
        unique_crops=unique_crops,
        total_markets=total_markets,
        total_districts=total_districts,
        top_crops=top_crops,
        env_data=env_data
    )

@bp.route("/crop_recommendation", methods=["GET", "POST"])
def show_crop_recommendation():
    try:
        # --- Soil input ---
        N = float(request.form.get("N") or 0.0)
        P = float(request.form.get("P") or 0.0)
        K = float(request.form.get("K") or 0.0)
        pH = float(request.form.get("pH") or 7.0)

        use_average_flag = bool(request.form.get("use_average_soil"))
        soil_district = request.form.get("soil_district") if use_average_flag else None

        # --- Weather input ---
        weather_option = request.form.get("weather_option")
        weather_district = request.form.get("weather_district")

        if weather_option == "api" and weather_district:
            weather_data = get_weather(weather_district)
            temperature = float(weather_data["temperature"])
            humidity = float(weather_data["humidity"])
            rainfall_input = float(weather_data["rainfall"])
        else:
            temperature = float(request.form.get("temperature") or 25.0)
            humidity = float(request.form.get("humidity") or 70.0)
            rainfall_input = float(request.form.get("rainfall") or 0.0)

        # --- Rainfall intensity adjustment ---
        intensity = request.form.get("rainfall_intensity") or "normal"
        multiplier = {"heavy": 90, "mild": 60, "normal": 30}.get(intensity.lower(), 30)
        final_rainfall = rainfall_input * multiplier  # only this is sent to model

        # --- Crop recommendation ---
        recommendations, input_summary = recommend_crops(
            N, P, K, pH, soil_district, use_average_flag,
            temperature, humidity, final_rainfall
        )

        # Add to input summary
        input_summary["Soil District"] = soil_district if use_average_flag else "Manual"
        input_summary["Weather District"] = weather_district if weather_option == "api" else "Manual"
        input_summary["Rainfall Intensity"] = intensity.capitalize()
        input_summary["Rainfall"] = rainfall_input  # original user/API input
        input_summary["Final Rainfall (mm)"] = final_rainfall  # multiplied value

        soil_districts = sorted(average_soil_df['district'].dropna().unique())
        weather_districts = get_all_districts()

        return render_template(
            "crop_recommendation.html",
            soil_districts=soil_districts,
            weather_districts=weather_districts,
            recommendation=recommendations,
            input_summary=input_summary,
            use_average_soil=use_average_flag,
            selected_soil_district=soil_district,
            selected_weather_district=weather_district,
            error=None
        )

    except Exception as e:
        soil_districts = sorted(average_soil_df['district'].dropna().unique())
        weather_districts = get_all_districts()
        return render_template(
            "crop_recommendation.html",
            soil_districts=soil_districts,
            weather_districts=weather_districts,
            recommendation=None,
            input_summary=None,
            use_average_soil=False,
            selected_soil_district=None,
            selected_weather_district=None,
            error=str(e)
        )

# AJAX endpoint for soil data
@bp.route("/get_soil/<district>")
def get_soil(district):
    soil = get_average_soil(district)
    if soil:
        return jsonify(soil)
    return jsonify({"error": "District not found"}), 404
