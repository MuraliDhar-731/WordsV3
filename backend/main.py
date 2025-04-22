
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from online_model import predict_difficulty, update_model

app = FastAPI()

words_df = pd.read_csv("data/realtime_words_dataset.csv")

class GameData(BaseModel):
    hints_used: int
    time_taken: float
    word_length: int
    word_frequency: float

@app.get("/start_game")
def start_game():
    row = words_df.sample(1).iloc[0]
    return {
        "word": "_" * int(row['word_length']),
        "length": int(row['word_length']),
        "frequency": float(row['word_frequency']),
        "true_word": row['word']
    }

@app.post("/predict_difficulty")
def predict_and_update(data: GameData):
    features = {
        "hints_used": data.hints_used,
        "time_taken": data.time_taken,
        "word_length": data.word_length,
        "word_frequency": data.word_frequency
    }

    predicted = predict_difficulty(features)

    true_score = (
        data.word_length * 0.5 +
        data.hints_used * 2 +
        data.time_taken * 0.1 +
        (1 - data.word_frequency) * 5
    )

    update_model(features, true_score)

    return {"predicted_difficulty": round(predicted, 2)}
