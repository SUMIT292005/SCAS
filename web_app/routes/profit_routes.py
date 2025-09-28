from flask import Blueprint, render_template, request, jsonify
import pandas as pd
from utils.api_market import get_price, get_states, get_districts, get_markets, get_commodities

profit_pest_routes = Blueprint("profit_pest_routes", __name__)

# ==========================================================
# Load CSV (for Manual profit estimation)
# ==========================================================
from pathlib import Path
 
# Build a relative path instead of hardcoding full path
Data = Path("data/processed/Price_Maharashtra1.csv")
 
# Convert to absolute path (works across systems)
Data = Data.resolve()
df = pd.read_csv(Data)

# ==========================================================
# Main Page
# ==========================================================
@profit_pest_routes.route("/profit_pest_estimation", methods=["GET"])
def profit_pest_page():
    districts = sorted(df["District Name"].dropna().unique())
    return render_template(
        "profit_pest_estimation.html",
        districts=districts,
        inputs={}
    )

# ==========================================================
# ------------------- MANUAL CSV-BASED SECTION -----------------------
# ==========================================================
@profit_pest_routes.route("/get_options_manual", methods=["POST"])
def get_options_manual():
    """
    Handles cascading dropdowns for manual CSV-based profit estimation.
    Returns either next-level options or prices for final level (Grade).
    """
    data = request.get_json()
    level = data.get("level")
    filters = data.get("filters", {})

    df_filtered = df.copy()
    # Apply all filters that have a value
    for key, value in filters.items():
        if value:
            df_filtered = df_filtered[df_filtered[key] == value]

    # Final level: return prices if Grade is selected
    if level == "Grade" and not df_filtered.empty:
        row = df_filtered.iloc[0]
        return jsonify({
            "min_price": float(row["Min Price (Rs./Quintal)"]),
            "max_price": float(row["Max Price (Rs./Quintal)"]),
            "modal_price": float(row["Modal Price (Rs./Quintal)"])
        })

    # Otherwise, return unique options for next dropdown
    next_col_map = {
        "District Name": "Market Name",
        "Market Name": "Commodity",
        "Commodity": "Variety",
        "Variety": "Grade"
    }
    next_col = next_col_map.get(level)
    if next_col and next_col in df_filtered.columns:
        options = sorted(df_filtered[next_col].dropna().unique().tolist())
    else:
        options = []

    return jsonify({"options": options})

# ==========================================================
# Optional legacy form submission (manual profit calculation)
# ==========================================================
@profit_pest_routes.route("/predict_profit", methods=["POST"])
def predict_profit():
    profit = None
    inputs = {}

    try:
        yield_val = float(request.form.get("Yield", 0))
        state = request.form.get("State")
        district = request.form.get("District")
        market = request.form.get("Market")
        crop = request.form.get("Crop")

        price_data = get_price(state, district, market, crop)

        if not price_data:
            price_input = request.form.get("Price")
            modal_price = float(price_input) if price_input else 0.0
        else:
            modal_price = price_data.get("modal_price", 0.0)

        profit = yield_val * modal_price

        inputs = {
            "Yield": yield_val,
            "State": state,
            "District": district,
            "Market": market,
            "Crop": crop,
            "Price": modal_price
        }

    except Exception as e:
        profit = f"Error: {str(e)}"

    return render_template(
        "profit_pest_estimation.html",
        profit=profit,
        inputs=inputs,
        districts=sorted(df["District Name"].dropna().unique())
    )

# ==========================================================
# ------------------- AUTO API-DRIVEN SECTION -----------------------
# ==========================================================
@profit_pest_routes.route("/auto_get_options", methods=["POST"])
def auto_get_options():
    data = request.get_json()
    level = data.get("level")
    filters = data.get("filters", {})

    if level == "state":
        return jsonify({"options": get_states()})
    elif level == "district":
        return jsonify({"options": get_districts(filters.get("state"))})
    elif level == "market":
        return jsonify({"options": get_markets(filters.get("state"), filters.get("district"))})
    elif level == "commodity":
        return jsonify({"options": get_commodities(filters.get("state"), filters.get("district"), filters.get("market"))})

    return jsonify({"options": []})

@profit_pest_routes.route("/auto_get_price", methods=["POST"])
def auto_get_price():
    data = request.get_json()
    state = data.get("state")
    district = data.get("district")
    market = data.get("market")
    commodity = data.get("commodity")

    price_data = get_price(state, district, market, commodity)
    return jsonify(price_data or {})
