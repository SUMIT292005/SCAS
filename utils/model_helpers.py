import pickle
import os

_model1 = None

def load_model1_crop_recommender():
    global _model1
    if _model1 is None:
        with open(os.path.join("models", "model1_crop_recommender.pkl"), "rb") as f:
            _model1 = pickle.load(f)
    return _model1
