import pandas as pd
import joblib
import pickle
import json
import os
# ---------------------------
# Crop Recommendation Setup
# ---------------------------


# Load crop recommendation model once
MODEL_PATH_CROP = os.path.join(os.path.dirname(__file__), '..', 'models', 'model1_crop_recommender.pkl')
_model1 = joblib.load(MODEL_PATH_CROP)

# Load average soil data once
SOIL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'average_data.xlsx')
average_soil_df = pd.read_excel(SOIL_PATH)

def get_average_soil(district):
    """
    Returns N, P, K, pH for the selected district from average_data.xlsx
    """
    row = average_soil_df[average_soil_df['district'] == district]
    if row.empty:
        return None
    return {
        'N': float(row['N'].values[0]),
        'P': float(row['P'].values[0]),
        'K': float(row['K'].values[0]),
        'pH': float(row['pH'].values[0])
    }

def recommend_crops(N, P, K, pH, district=None, use_average=False,
                    temperature=None, humidity=None, rainfall=None, model=_model1):
    """
    Prepare input features for the model.
    Handles manual entries for soil and weather values.
    If use_average is True, NPK and pH are fetched from the selected district.
    """
    # If using average soil, fetch values from Excel
    if use_average and district:
        soil = get_average_soil(district)
        if soil:
            N = soil['N']
            P = soil['P']
            K = soil['K']
            pH = soil['pH']

    # Convert inputs to floats
    N = float(N)
    P = float(P)
    K = float(K)
    pH = float(pH)

    # Use provided weather values or defaults
    temperature = float(temperature) if temperature is not None else 25.0
    humidity = float(humidity) if humidity is not None else 70.0
    rainfall = float(rainfall) if rainfall is not None else 150.0

    # Create feature list for model
    features = [[N, P, K, temperature, humidity, pH, rainfall]]

    # Predict probabilities
    probabilities = model.predict_proba(features)[0]
    crop_indexes = probabilities.argsort()[::-1][:3]
    all_labels = model.classes_
    recommendations = [(all_labels[i], float(probabilities[i])) for i in crop_indexes]

    # Input summary to show back on UI
    input_summary = {
        "N": N,
        "P": P,
        "K": K,
        "pH": pH,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall
    }

    return recommendations, input_summary


# ---------------------------
# Yield Prediction Setup
# ---------------------------

# Path to saved yield prediction model
MODEL_PATH_YIELD = os.path.join(os.path.dirname(__file__), '..', 'models', 'crop_yield_predictor.pkl')

def predict_yield(input_data: dict) -> float:
    """
    Predict crop yield (tons/ha) from input data.

    Parameters:
        input_data (dict): Dictionary with keys matching training features:
            - Region, Soil_Type, Crop, Weather_Condition
            - Rainfall_mm, Temperature_Celsius, Days_to_Harvest
            - Fertilizer_Used, Irrigation_Used

    Returns:
        float: Predicted yield in tons per hectare
    """
    # Load saved model
    with open(MODEL_PATH_YIELD, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoders = model_data["label_encoders"]
    feature_names = model_data["feature_names"]

    # Columns info
    categorical_cols = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
    bool_cols = ["Fertilizer_Used", "Irrigation_Used"]
    numerical_cols = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col in categorical_cols:
        if col in input_df.columns:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0  # Default for unseen labels

    # Convert boolean columns to int
    for col in bool_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(int)

    # Scale numerical columns
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Make prediction
    prediction = model.predict(input_df)[0]
    return prediction









from utils.fertilizer_rules import get_growth_fertilizer

def suggest_growth_fertilizer(crop, N, P, K, pH, rainfall=None, temperature=None, humidity=None):
    """
    Wrapper function to process input from web form and return
    fertilizer list + weather-based warnings.
    """
    try:
        # Required soil inputs
        N, P, K, pH = float(N), float(P), float(K), float(pH)

        # Optional weather inputs (only if provided)
        if rainfall is not None and rainfall != "":
            rainfall = float(rainfall)
        else:
            rainfall = None

        if temperature is not None and temperature != "":
            temperature = float(temperature)
        else:
            temperature = None

        if humidity is not None and humidity != "":
            humidity = float(humidity)
        else:
            humidity = None

        # Call fertilizer rules engine
        return get_growth_fertilizer(
            crop, N, P, K, pH,
            rainfall=rainfall,
            temperature=temperature,
            humidity=humidity
        )

    except ValueError:
        return {
            "fertilizers": [],
            "warnings": ["Invalid input values. Please enter numeric values for N, P, K, and pH."]
        }




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import json
import tensorflow as tf  # only needed for preprocess_input

# ============================================
# TFLITE MODELS (SUPER LIGHTWEIGHT)
# ============================================
CROP_MODELS = {
    "cotton": (
        "models/cotton_model.tflite",
        "models/cotton_class_indices.json",
        "rescale",
        False
    ),
    "tomato": (
        "models/tomato_model.tflite",
        "models/tomato_class_indices.json",
        "efficientnet",
        False
    ),
}

loaded_interpreters = {}
loaded_labels = {}

# ============================================
# LOAD TFLITE MODEL + LABELS
# ============================================
def load_model_and_labels(crop):
    if crop not in loaded_interpreters:
        model_path, label_path, _, _ = CROP_MODELS[crop]

        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        loaded_interpreters[crop] = interpreter

        # Load labels (supports BOTH dict and list)
        with open(label_path, "r") as f:
            labels_dict = json.load(f)

            # Case 1: labels are already a list → use directly
            if isinstance(labels_dict, list):
                loaded_labels[crop] = labels_dict

            # Case 2: labels are a dict → convert to list (sorted by index)
            elif isinstance(labels_dict, dict):
                sorted_labels = [None] * len(labels_dict)
                for k, v in labels_dict.items():
                    sorted_labels[int(v)] = k
                loaded_labels[crop] = sorted_labels

            else:
                raise ValueError(f"❌ Unknown label format in {label_path}")

    return loaded_interpreters[crop], loaded_labels[crop]


# ============================================
# IMAGE PREPROCESS
# ============================================
def preprocess_image(img_path, mode):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    if mode == "rescale":
        img_array = img_array / 255.0
    elif mode == "efficientnet":
        img_array = eff_preprocess(img_array)

    return img_array.astype(np.float32)

# ============================================
# PREDICT USING TFLITE
# ============================================
def detect_disease_from_image(img_path, crop):
    interpreter, class_labels = load_model_and_labels(crop)
    _, _, preprocess_mode, _ = CROP_MODELS[crop]

    img_array = preprocess_image(img_path, preprocess_mode)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    class_index = np.argmax(preds)
    predicted_class = class_labels[class_index]
    confidence = round(float(np.max(preds) * 100), 2)

    return f"Predicted Class: {predicted_class}", f"Confidence: {confidence}%"

def recommend_treatment(crop, disease):
    """
    Fetch treatment recommendation for given crop and disease
    from model7_treatment_advisor_rules.json (case-insensitive lookup).
    """
    file_path = os.path.join("models", "model7_treatment_advisor_rules.json")

    if not os.path.exists(file_path):
        return {
            "Chemical": ["N/A"],
            "Organic": ["N/A"],
            "Cultural": ["N/A"],
            "Message": f"⚠️ File not found at {file_path}"
        }

    with open(file_path, "r") as f:
        data = json.load(f)

    crop = crop.strip().lower()
    disease = disease.strip().lower()

    for c_name, diseases in data.items():
        if c_name.lower() == crop:
            for d_name, rec in diseases.items():
                if d_name.lower() == disease:
                    # Handle Healthy case
                    if "Message" in rec:
                        return {"Message": rec["Message"]}
                    return rec

    # If not found
    return {
        "Chemical": ["Unknown"],
        "Organic": ["Unknown"],
        "Cultural": ["Unknown"],
        "Message": f"No treatment found for crop='{crop}', disease='{disease}'"
    }