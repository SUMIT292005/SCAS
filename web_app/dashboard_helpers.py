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

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import os, json

# ===========================================
# Cache for loaded models and labels
# ===========================================
loaded_models = {}
loaded_labels = {}

# ===========================================
# Model + label mapping (MAIZE REMOVED)
# ===========================================
CROP_MODELS = {
    "cotton": (
        "models/updated_cotton_disease_efficientnetb3.h5",
        "models/cotton_class_indices.json",
        "rescale",        # trained with /255.0
        False             # preprocessing not inside model
    ),
    "tomato": (
        "models/updated_tomato_disease_efficientnetb0.h5",
        "models/tomato_class_indices.json",
        "efficientnet",   # trained with eff_preprocess
        False
    ),
    "soybean": (
        "models/updated_soybean_disease_efficientnetb3.h5",
        "models/soybean_class_indices.json",
        "rescale",
        False
    )
}

# ===========================================
# Load model + labels
# ===========================================
def load_model_and_labels(crop):
    """Load model and its class labels for a given crop."""
    if crop not in loaded_models:
        model_path, label_path, _, _ = CROP_MODELS[crop]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found for {crop}: {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ Class label file missing for {crop}: {label_path}")

        # For problematic models, use compile=False
        if crop in ["cotton"]:
            print(f"🔄 Loading {crop} model with compile=False...")
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            model = tf.keras.models.load_model(model_path)
        
        print(f"🔍 Loaded {crop} model with input shape: {model.input_shape}")
        loaded_models[crop] = model

        # Load labels
        with open(label_path, "r") as f:
            labels_dict = json.load(f)
            if isinstance(labels_dict, dict):
                sorted_labels = [None] * len(labels_dict)
                for k, v in labels_dict.items():
                    sorted_labels[int(v)] = k
                loaded_labels[crop] = sorted_labels
            else:
                loaded_labels[crop] = labels_dict

    return loaded_models[crop], loaded_labels[crop]

# ===========================================
# Preprocess image
# ===========================================
def preprocess_image(img_path, mode, internal_preprocessing, model):
    """Apply preprocessing with automatic shape detection."""
    
    input_shape = model.input_shape
    expected_channels = input_shape[-1]

    color_mode = "grayscale" if expected_channels == 1 else "rgb"
    
    img = image.load_img(img_path, target_size=(224, 224), color_mode=color_mode)
    img_array = image.img_to_array(img)

    if expected_channels == 1 and img_array.shape[-1] == 3:
        img_array = np.mean(img_array, axis=-1, keepdims=True)
    elif expected_channels == 3 and len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    if internal_preprocessing:
        return img_array
    else:
        if mode == "rescale":
            return img_array / 255.0
        elif mode == "efficientnet":
            return eff_preprocess(img_array)
        else:
            raise ValueError(f"❌ Unknown preprocessing mode: {mode}")

# ===========================================
# Predict
# ===========================================
def detect_disease_from_image(img_path, crop):
    model, class_labels = load_model_and_labels(crop)
    _, _, preprocess_mode, internal_preprocessing = CROP_MODELS[crop]

    img_array = preprocess_image(img_path, preprocess_mode, internal_preprocessing, model)

    preds = model.predict(img_array, verbose=0)
    class_index = np.argmax(preds[0])
    predicted_class = class_labels[class_index]
    confidence = round(float(np.max(preds[0]) * 100), 2)

    return f"✅ Predicted Class: {predicted_class}", f"✅ Confidence: {confidence:.2f}%"

# ===========================================
# 🔄 PRELOAD ALL CROP DISEASE MODELS AT STARTUP
# ===========================================
print("🔄 Preloading disease detection models...")

for c in CROP_MODELS.keys():
    try:
        load_model_and_labels(c)
        print(f"✅ Loaded {c} model successfully")
    except Exception as e:
        print(f"❌ Failed to load {c} model: {e}")

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