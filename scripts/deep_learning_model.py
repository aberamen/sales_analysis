import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def train_lstm(data):
    """Train an LSTM model."""
    try:
        # Prepare time series data
        sales = data['Sales'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        sales_scaled = scaler.fit_transform(sales)

        # Create supervised learning data (sliding window)
        X, y = [], []
        window_size = 30
        for i in range(len(sales_scaled) - window_size):
            X.append(sales_scaled[i:i + window_size])
            y.append(sales_scaled[i + window_size])

        X, y = np.array(X), np.array(y)

        # Build LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(window_size, 1)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32)

        # Save model
        model.save(f'../data/models/lstm-{timestamp}.h5')
    except Exception as e:
        logging.error(f"Error training LSTM model: {e}")
        raise
