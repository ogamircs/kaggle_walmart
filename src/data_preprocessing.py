"""
Data loading and preprocessing for Walmart Sales Forecasting.
"""
import pandas as pd
import numpy as np
from typing import Tuple
from .config import DATA_PATH, VALIDATION_START_DATE


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load and preprocess the Walmart sales data.

    Args:
        filepath: Path to the CSV file. Uses default if None.

    Returns:
        Preprocessed DataFrame sorted by Store and Date.
    """
    if filepath is None:
        filepath = DATA_PATH

    df = pd.read_csv(filepath)

    # Convert Date from DD-MM-YYYY to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Sort by Store and Date for time series consistency
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate data integrity and return summary statistics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with validation results
    """
    validation = {
        'total_rows': len(df),
        'n_stores': df['Store'].nunique(),
        'n_weeks': df['Date'].nunique(),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'negative_sales': (df['Weekly_Sales'] < 0).sum()
    }

    # Check for complete store coverage
    expected_rows = validation['n_stores'] * validation['n_weeks']
    validation['complete_coverage'] = validation['total_rows'] == expected_rows

    return validation


def time_based_split(
    df: pd.DataFrame,
    val_start_date: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by time for proper time series validation.

    Args:
        df: Input DataFrame with Date column
        val_start_date: Start date for validation set (YYYY-MM-DD format)

    Returns:
        Tuple of (train_df, val_df)
    """
    if val_start_date is None:
        val_start_date = VALIDATION_START_DATE

    val_start = pd.to_datetime(val_start_date)

    train_df = df[df['Date'] < val_start].copy()
    val_df = df[df['Date'] >= val_start].copy()

    print(f"Training data: {len(train_df)} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Validation data: {len(val_df)} rows ({val_df['Date'].min()} to {val_df['Date'].max()})")

    return train_df, val_df


def get_store_data(df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    """
    Extract data for a single store.

    Args:
        df: Input DataFrame
        store_id: Store identifier (1-45)

    Returns:
        DataFrame filtered for the specified store
    """
    return df[df['Store'] == store_id].copy()


def prepare_prophet_data(df: pd.DataFrame, store_id: int = None) -> pd.DataFrame:
    """
    Prepare data in Prophet format (ds, y columns).

    Args:
        df: Input DataFrame
        store_id: Optional store filter

    Returns:
        DataFrame with 'ds' (dates) and 'y' (target) columns
    """
    if store_id is not None:
        df = df[df['Store'] == store_id].copy()

    prophet_df = df[['Date', 'Weekly_Sales']].copy()
    prophet_df.columns = ['ds', 'y']

    # Add regressors if available
    for col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']:
        if col in df.columns:
            prophet_df[col] = df[col].values

    return prophet_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'shape': df.shape,
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d'),
            'n_weeks': df['Date'].nunique()
        },
        'stores': {
            'count': df['Store'].nunique(),
            'list': sorted(df['Store'].unique().tolist())
        },
        'weekly_sales': {
            'mean': df['Weekly_Sales'].mean(),
            'std': df['Weekly_Sales'].std(),
            'min': df['Weekly_Sales'].min(),
            'max': df['Weekly_Sales'].max(),
            'median': df['Weekly_Sales'].median()
        },
        'holiday_weeks': {
            'count': df['Holiday_Flag'].sum(),
            'percentage': df['Holiday_Flag'].mean() * 100
        }
    }

    return summary


if __name__ == "__main__":
    # Test data loading
    df = load_data()
    print("\nData Validation:")
    validation = validate_data(df)
    for key, value in validation.items():
        print(f"  {key}: {value}")

    print("\nData Summary:")
    summary = get_data_summary(df)
    print(f"  Shape: {summary['shape']}")
    print(f"  Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"  Stores: {summary['stores']['count']}")
    print(f"  Weekly Sales Mean: ${summary['weekly_sales']['mean']:,.2f}")

    # Test split
    print("\nTrain/Validation Split:")
    train_df, val_df = time_based_split(df)
