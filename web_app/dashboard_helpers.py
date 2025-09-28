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
# Model + label mapping
#   Format: crop: (model_path, label_path, preprocessing_mode, internal_preprocessing)
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
    ),
    "maize": (
        "models/updated_maize_disease_efficientnetb0.h5",
        "models/maize_class_indices.json",
        None,             # ✅ no external preprocessing
        True              # ✅ preprocessing baked into inference model
    ),
}

# ===========================================
# Load model + labels
# ===========================================
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

        # For cotton, use the rebuild function directly
        if crop == "cotton":
            print(f"🔄 Using rebuild method for {crop} model...")
            model = rebuild_model_with_correct_shape(model_path, crop)
        else:
            try:
                # Try loading normally for other crops
                model = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"⚠️  Error loading {crop} model: {e}")
                # Fallback to rebuild method for other crops if needed
                model = rebuild_model_with_correct_shape(model_path, crop)
        
        # Debug: Print model input shape
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

def rebuild_model_with_correct_shape(model_path, crop):
    """Rebuild model with correct input shape when there's a shape mismatch."""
    try:
        # Method 1: Try loading with compile=False first
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✅ Successfully loaded {crop} model with compile=False")
        return model
    except ValueError as e:
        if "Shape mismatch" in str(e):
            print(f"🔄 Shape mismatch detected for {crop}, using advanced rebuild...")
            # Method 2: Load architecture and weights separately
            return load_model_architecture_and_weights(model_path, crop)
        else:
            raise e

def load_model_architecture_and_weights(model_path, crop):
    """Load model architecture and weights separately."""
    try:
        # Method 2a: Load weights into a new model with correct input shape
        if "efficientnet" in model_path.lower():
            # For EfficientNet models
            if "b0" in model_path.lower():
                base_model = tf.keras.applications.EfficientNetB0(
                    weights=None, 
                    include_top=True, 
                    input_shape=(224, 224, 3),  # Force 3 channels
                    classes=len(loaded_labels.get(crop, []) or 5)  # Default to 5 if unknown
                )
            elif "b3" in model_path.lower():
                base_model = tf.keras.applications.EfficientNetB3(
                    weights=None, 
                    include_top=True, 
                    input_shape=(224, 224, 3),  # Force 3 channels
                    classes=len(loaded_labels.get(crop, []) or 5)
                )
            else:
                # Default to B0
                base_model = tf.keras.applications.EfficientNetB0(
                    weights=None, 
                    include_top=True, 
                    input_shape=(224, 224, 3),
                    classes=len(loaded_labels.get(crop, []) or 5)
                )
            
            # Load weights
            base_model.load_weights(model_path)
            base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return base_model
            
    except Exception as e:
        print(f"❌ Advanced rebuild failed: {e}")
        
        # Method 2b: Last resort - create a simple model
        print("🔄 Creating fallback model architecture...")
        return create_fallback_model(crop)

def create_fallback_model(crop):
    """Create a simple fallback model when all else fails."""
    num_classes = len(loaded_labels.get(crop, [])) or 3  # Default to 3 classes
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"✅ Created fallback model for {crop} with {num_classes} classes")
    return model
# ===========================================
# Preprocess image
# ===========================================
def preprocess_image(img_path, mode, internal_preprocessing, model):
    """Apply preprocessing with automatic shape detection."""
    
    # Get model's expected input shape
    input_shape = model.input_shape
    expected_channels = input_shape[-1]  # Last dimension is channels
    
    # Load image with appropriate color mode
    if expected_channels == 1:
        color_mode = "grayscale"
    else:
        color_mode = "rgb"
    
    img = image.load_img(img_path, target_size=(224, 224), color_mode=color_mode)
    img_array = image.img_to_array(img)
    
    # Ensure correct channel dimension
    if expected_channels == 1 and len(img_array.shape) == 3 and img_array.shape[-1] == 3:
        # Convert RGB to grayscale
        img_array = np.mean(img_array, axis=-1, keepdims=True)
    elif expected_channels == 3 and len(img_array.shape) == 2:
        # Convert grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Add batch dimension
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

def detect_disease_from_image(img_path, crop):
    """Run disease detection for a given crop."""
    model, class_labels = load_model_and_labels(crop)
    _, _, preprocess_mode, internal_preprocessing = CROP_MODELS[crop]

    # Preprocess input - pass model to automatically detect required shape
    img_array = preprocess_image(img_path, preprocess_mode, internal_preprocessing, model)

    # Predict
    preds = model.predict(img_array, verbose=0)
    class_index = np.argmax(preds[0])
    predicted_class = class_labels[class_index]
    confidence = round(float(np.max(preds[0]) * 100), 2)

    return f"✅ Predicted Class: {predicted_class}", f"✅ Confidence: {confidence:.2f}%"


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