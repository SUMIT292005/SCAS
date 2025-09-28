from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from web_app.dashboard_helpers import predict_yield
from utils.api_weather import get_weather, get_all_districts

yield_routes = Blueprint("yield_routes", __name__)

# Load state-region-soil dataset once
state_data = pd.read_csv("data/processed/State_Region_SoilType.csv")

# Unique states for dropdown
STATES = sorted(state_data["State"].dropna().unique())

# Convert dataset to dict for JS auto-fill
STATE_REGION_SOIL = state_data[['State', 'Region', 'Soil_Type']].dropna().to_dict(orient='records')

# All districts for weather dropdown
DISTRICTS = get_all_districts()

# ----------------------------
# AJAX endpoint to fetch weather
# ----------------------------
@yield_routes.route("/get_weather/<district>")
def fetch_weather(district):
    try:
        weather = get_weather(district)
        return jsonify(weather)
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------
# Yield Prediction Route
# ----------------------------
@yield_routes.route("/yield_prediction", methods=["GET", "POST"])
def yield_prediction_route():
    prediction = None
    input_data = {}

    if request.method == "POST":
        try:
            # ----- Region & Soil -----
            auto_fill_region = request.form.get("auto_fill") == "on"
            selected_state = request.form.get("State")
            if auto_fill_region and selected_state:
                row = state_data[state_data["State"] == selected_state].iloc[0]
                region = row["Region"]
                soil_type = row["Soil_Type"]
            else:
                region = request.form.get("Region")
                soil_type = request.form.get("Soil_Type")

            # ----- Weather -----
            auto_fill_weather = request.form.get("auto_weather") == "on"
            if auto_fill_weather:
                district = request.form.get("District")
                if district:
                    try:
                        weather_data = get_weather(district)
                        temperature = weather_data.get("temperature", 0.0)
                        rainfall = weather_data.get("rainfall", 0.0)
                    except Exception:
                        temperature = float(request.form.get("Temperature_Celsius", 0.0))
                        rainfall = float(request.form.get("Rainfall_mm", 0.0))
                else:
                    temperature = float(request.form.get("Temperature_Celsius", 0.0))
                    rainfall = float(request.form.get("Rainfall_mm", 0.0))
            else:
                temperature = float(request.form.get("Temperature_Celsius", 0.0))
                rainfall = float(request.form.get("Rainfall_mm", 0.0))

            # ----- Collect other form data -----
            input_data = {
                "State": selected_state,
                "Region": region,
                "Soil_Type": soil_type,
                "Crop": request.form.get("Crop"),
                "Rainfall_mm": rainfall,
                "Temperature_Celsius": temperature,
                "Fertilizer_Used": request.form.get("Fertilizer_Used") == "True",
                "Irrigation_Used": request.form.get("Irrigation_Used") == "True",
                "Weather_Condition": request.form.get("Weather_Condition"),
                "Days_to_Harvest": int(request.form.get("Days_to_Harvest", 0)),
            }

            # Call prediction helper
            prediction = predict_yield(input_data)
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error in prediction: {str(e)}"

    return render_template(
        "yield_prediction.html",
        prediction=prediction,
        states=STATES,
        state_region_soil=STATE_REGION_SOIL,
        districts=DISTRICTS,
        input_data=input_data
    )
