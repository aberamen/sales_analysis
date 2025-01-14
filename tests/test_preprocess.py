import pytest
from scripts.preprocess import load_data, clean_data
import pandas as pd

def test_load_data():
    """Test loading data."""
    data = load_data('../data/raw/store_sales.csv')
    assert not data.empty
    assert 'Date' in data.columns

def test_clean_data():
    """Test cleaning data."""
    data = pd.DataFrame({
        "Sales": [100, 200, 300, 5000],  # Outlier
        "CompetitionDistance": [100, None, 300, 400]
    })
    cleaned = clean_data(data)
    assert cleaned['CompetitionDistance'].isna().sum() == 0
    assert cleaned['Sales'].max() <= 1000  # Example upper limit
