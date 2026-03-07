"""
Pytest configuration and shared fixtures for Walmart Sales Forecasting tests.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_walmart_data():
    """
    Create a sample DataFrame mimicking the Walmart dataset structure.
    This fixture creates synthetic data that matches the expected format.
    """
    np.random.seed(42)

    # Generate dates for 2 years of weekly data
    dates = pd.date_range(start='2010-01-01', end='2011-12-31', freq='W')
    n_stores = 5
    n_dates = len(dates)

    data = []
    for store in range(1, n_stores + 1):
        for date in dates:
            data.append({
                'Store': store,
                'Date': date,
                'Weekly_Sales': np.random.uniform(500000, 2000000),
                'Holiday_Flag': 1 if np.random.random() < 0.1 else 0,
                'Temperature': np.random.uniform(30, 90),
                'Fuel_Price': np.random.uniform(2.5, 4.0),
                'CPI': np.random.uniform(180, 230),
                'Unemployment': np.random.uniform(5, 12)
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df


@pytest.fixture
def small_sample_data():
    """
    Create a small sample DataFrame for quick tests.
    """
    np.random.seed(42)

    dates = pd.date_range(start='2010-01-01', periods=20, freq='W')
    n_stores = 2

    data = []
    for store in range(1, n_stores + 1):
        for date in dates:
            data.append({
                'Store': store,
                'Date': date,
                'Weekly_Sales': np.random.uniform(500000, 2000000),
                'Holiday_Flag': 1 if np.random.random() < 0.1 else 0,
                'Temperature': np.random.uniform(30, 90),
                'Fuel_Price': np.random.uniform(2.5, 4.0),
                'CPI': np.random.uniform(180, 230),
                'Unemployment': np.random.uniform(5, 12)
            })

    df = pd.DataFrame(data)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df


@pytest.fixture
def sample_train_val_arrays():
    """
    Create sample training and validation arrays for model testing.
    """
    np.random.seed(42)
    n_train = 160
    n_val = 40
    n_features = 10

    X_train = np.random.randn(n_train, n_features)
    y_train = X_train[:, 0] * 100000 + np.random.randn(n_train) * 10000 + 1000000

    X_val = np.random.randn(n_val, n_features)
    y_val = X_val[:, 0] * 100000 + np.random.randn(n_val) * 10000 + 1000000

    return X_train, y_train, X_val, y_val


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
