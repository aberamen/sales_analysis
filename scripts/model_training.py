from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

def train_model(data, output_model_path):
    """Train a regression model and save it."""
    try:
        # Split data
        X = data.drop(['Sales', 'Date'], axis=1)
        y = data['Sales']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define pipeline
        pipeline = Pipeline([
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Save model
        joblib.dump(pipeline, output_model_path)
        logging.info(f"Model saved to {output_model_path}.")
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
def analyze_model(pipeline, X_test):
    """Analyze feature importance and confidence intervals."""
    try:
        # Feature importance
        feature_importances = pipeline.named_steps['model'].feature_importances_
        logging.info(f"Feature importances: {feature_importances}")

        # Confidence intervals (basic bootstrap approach)
        preds = pipeline.predict(X_test)
        lower = np.percentile(preds, 2.5)
        upper = np.percentile(preds, 97.5)
        logging.info(f"Prediction confidence interval: [{lower}, {upper}]")
    except Exception as e:
        logging.error(f"Error in model analysis: {e}")
        raise
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
output_model_path = f'../data/models/random_forest-{timestamp}.pkl'

if __name__ == "__main__":
    data = pd.read_csv('../data/processed/cleaned_data.csv')
    train_model(data, '../data/models/random_forest.pkl')
