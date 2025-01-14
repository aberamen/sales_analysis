import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    filename='../customer_behavior_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path):
    """Load dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'])
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_data(data):
    """Handle missing values and outliers."""
    try:
        # Handle missing values
        data['CompetitionDistance'].fillna(data['CompetitionDistance'].median(), inplace=True)
        data['PromoInterval'].fillna('None', inplace=True)

        # Outlier detection and removal for Sales
        q1 = data['Sales'].quantile(0.25)
        q3 = data['Sales'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data['Sales'] >= lower_bound) & (data['Sales'] <= upper_bound)]

        logging.info("Data cleaned successfully.")
        return data
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise

if __name__ == "__main__":
    file_path = '../data/raw/store_sales.csv'
    output_path = '../data/processed/cleaned_data.csv'

    data = load_data(file_path)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path}.")
