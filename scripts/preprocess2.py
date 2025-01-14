from sklearn.preprocessing import StandardScaler
import numpy as np

def feature_engineering(data):
    """Generate new features from datetime columns."""
    try:
        # Extract temporal features
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        data['IsMonthStart'] = data['Date'].dt.is_month_start.astype(int)
        data['IsMonthEnd'] = data['Date'].dt.is_month_end.astype(int)

        # Feature: Days to next holiday (assume fixed dates for simplicity)
        holidays = ['2024-01-01', '2024-12-25']  # Example dates
        data['DaysToHoliday'] = data['Date'].apply(
            lambda x: min([(pd.to_datetime(h) - x).days for h in holidays if h > str(x)] + [np.inf])
        )

        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['Sales', 'Customers', 'CompetitionDistance']
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        logging.info("Feature engineering completed.")
        return data
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        raise
