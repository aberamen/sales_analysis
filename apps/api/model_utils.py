import joblib
import tensorflow as tf
import numpy as np

# Load Random Forest model
def load_random_forest_model(model_path='../data/models/random_forest.pkl'):
    return joblib.load(model_path)

# Load LSTM model
def load_lstm_model(model_path='../data/models/lstm_model.h5'):
    return tf.keras.models.load_model(model_path)

# Make predictions with Random Forest
def predict_random_forest(model, input_data):
    return model.predict(input_data).tolist()

# Make predictions with LSTM
def predict_lstm(model, input_data, scaler):
    input_data = scaler.transform(input_data)
    input_data = np.array(input_data).reshape((1, input_data.shape[0], 1))
    prediction = model.predict(input_data)
    return scaler.inverse_transform(prediction).flatten().tolist()
