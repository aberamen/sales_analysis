from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from model_utils import load_random_forest_model, load_lstm_model, predict_random_forest, predict_lstm

# Initialize FastAPI app
app = FastAPI()

# Load models
rf_model = load_random_forest_model()
lstm_model = load_lstm_model()

# Define input schema
class PredictionInput(BaseModel):
    model_type: str  # "random_forest" or "lstm"
    input_data: list  # Input data for the model

# Define API endpoints
@app.post("/predict/")
async def predict(input_data: PredictionInput):
    try:
        if input_data.model_type == "random_forest":
            # Convert input data to DataFrame
            data = pd.DataFrame(input_data.input_data)
            predictions = predict_random_forest(rf_model, data)
            return {"model": "Random Forest", "predictions": predictions}
        
        elif input_data.model_type == "lstm":
            scaler_path = '../data/models/scaler.pkl'  # Replace with actual scaler path
            scaler = joblib.load(scaler_path)
            predictions = predict_lstm(lstm_model, np.array(input_data.input_data), scaler)
            return {"model": "LSTM", "predictions": predictions}
        
        else:
            raise ValueError("Invalid model type. Choose 'random_forest' or 'lstm'.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the Sales Prediction API!"}
