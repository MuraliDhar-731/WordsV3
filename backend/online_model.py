
from river import linear_model, preprocessing
import pickle
import os

model_path = "models/online_model.pkl"

# Create pipeline
model = preprocessing.StandardScaler() | linear_model.LinearRegression()

# Load if exists
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# Predict function
def predict_difficulty(features: dict) -> float:
    return model.predict_one(features)

# Update model live
def update_model(features: dict, true_score: float):
    global model
    prediction = model.predict_one(features)
    model.learn_one(features, true_score)
    print(f"[Online Learning] Predicted: {round(prediction, 2)}, True: {round(true_score, 2)}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
