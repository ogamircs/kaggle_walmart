"""
Feature engineering for Walmart Sales Forecasting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .config import HOLIDAYS, ALL_HOLIDAY_DATES, TARGET


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the Date column.

    Args:
        df: DataFrame with Date column

    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['quarter'] = df['Date'].dt.quarter

    # Week of month (1-5)
    df['week_of_month'] = (df['Date'].dt.day - 1) // 7 + 1

    # Boolean indicators
    df['is_month_start'] = (df['Date'].dt.day <= 7).astype(int)
    df['is_month_end'] = (df['Date'].dt.day >= 24).astype(int)
    df['is_quarter_start'] = (df['month'].isin([1, 4, 7, 10]) & (df['Date'].dt.day <= 7)).astype(int)
    df['is_quarter_end'] = (df['month'].isin([3, 6, 9, 12]) & (df['Date'].dt.day >= 24)).astype(int)
    df['is_year_start'] = ((df['month'] == 1) & (df['Date'].dt.day <= 7)).astype(int)
    df['is_year_end'] = ((df['month'] == 12) & (df['Date'].dt.day >= 24)).astype(int)

    return df


def create_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical (sine/cosine) encoding for periodic features.

    Args:
        df: DataFrame with temporal features

    Returns:
        DataFrame with cyclical features added
    """
    df = df.copy()

    # Month cyclical encoding (period = 12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Week cyclical encoding (period = 52)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Day of year cyclical encoding (period = 365)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Quarter cyclical encoding (period = 4)
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

    return df


def create_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create holiday-related features.

    Args:
        df: DataFrame with Date column

    Returns:
        DataFrame with holiday features added
    """
    df = df.copy()

    # Specific holiday indicators (check if date falls within holiday week)
    for holiday_name, holiday_dates in HOLIDAYS.items():
        # Mark the week containing the holiday
        col_name = f'is_{holiday_name.lower()}'
        df[col_name] = 0
        for hdate in holiday_dates:
            # Mark weeks within 3 days of the holiday
            mask = (df['Date'] >= hdate - pd.Timedelta(days=3)) & \
                   (df['Date'] <= hdate + pd.Timedelta(days=3))
            df.loc[mask, col_name] = 1

    # Pre-holiday indicators (2 weeks before)
    df['is_pre_thanksgiving'] = 0
    df['is_pre_christmas'] = 0
    for tg_date in HOLIDAYS['Thanksgiving']:
        mask = (df['Date'] >= tg_date - pd.Timedelta(weeks=2)) & \
               (df['Date'] < tg_date - pd.Timedelta(days=3))
        df.loc[mask, 'is_pre_thanksgiving'] = 1

    for xmas_date in HOLIDAYS['Christmas']:
        mask = (df['Date'] >= xmas_date - pd.Timedelta(weeks=4)) & \
               (df['Date'] < xmas_date - pd.Timedelta(days=3))
        df.loc[mask, 'is_pre_christmas'] = 1

    # Post-holiday indicator (Black Friday week - week after Thanksgiving)
    df['is_black_friday_week'] = 0
    for tg_date in HOLIDAYS['Thanksgiving']:
        mask = (df['Date'] > tg_date) & \
               (df['Date'] <= tg_date + pd.Timedelta(weeks=1))
        df.loc[mask, 'is_black_friday_week'] = 1

    # Distance to nearest holiday (in weeks)
    df['weeks_to_next_holiday'] = df['Date'].apply(
        lambda x: min([(h - x).days for h in ALL_HOLIDAY_DATES if h >= x], default=52) // 7
    )
    df['weeks_since_last_holiday'] = df['Date'].apply(
        lambda x: min([(x - h).days for h in ALL_HOLIDAY_DATES if h <= x], default=52) // 7
    )

    return df


def create_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Create lag features for sales (per store).

    Args:
        df: DataFrame with Store and Weekly_Sales columns
        lags: List of lag periods (in weeks)

    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 2, 4, 8, 12, 52]

    df = df.copy()

    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby('Store')[TARGET].shift(lag)

    # Same week last year (using week of year)
    df['sales_same_week_last_year'] = df.groupby(['Store', 'week_of_year'])[TARGET].shift(1)

    return df


def create_rolling_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """
    Create rolling window statistics (per store).

    Args:
        df: DataFrame with Store and Weekly_Sales columns
        windows: List of window sizes (in weeks)

    Returns:
        DataFrame with rolling features added
    """
    if windows is None:
        windows = [4, 8, 12, 26]

    df = df.copy()

    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}w'] = df.groupby('Store')[TARGET].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Rolling std
        df[f'rolling_std_{window}w'] = df.groupby('Store')[TARGET].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
        )

        # Rolling min/max for smaller windows
        if window <= 8:
            df[f'rolling_min_{window}w'] = df.groupby('Store')[TARGET].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )
            df[f'rolling_max_{window}w'] = df.groupby('Store')[TARGET].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )

    # Exponentially weighted moving average
    for span in [4, 8, 12]:
        df[f'ewma_{span}w'] = df.groupby('Store')[TARGET].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )

    return df


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trend and momentum features.

    Args:
        df: DataFrame with lag features

    Returns:
        DataFrame with trend features added
    """
    df = df.copy()

    # Sales differences
    df['sales_diff_1'] = df.groupby('Store')[TARGET].diff(1)
    df['sales_diff_4'] = df.groupby('Store')[TARGET].diff(4)

    # Percent changes
    df['sales_pct_change_1'] = df.groupby('Store')[TARGET].pct_change(1)
    df['sales_pct_change_4'] = df.groupby('Store')[TARGET].pct_change(4)

    # Trend indicators
    if 'rolling_mean_4w' in df.columns:
        df['above_rolling_mean_4w'] = (df[TARGET] > df['rolling_mean_4w']).astype(int)

    if 'rolling_std_4w' in df.columns and 'rolling_mean_4w' in df.columns:
        df['trend_strength'] = (df[TARGET] - df['rolling_mean_4w']) / (df['rolling_std_4w'] + 1)

    return df


def create_store_features(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create store-level aggregate features.
    IMPORTANT: Use only training data statistics to prevent leakage.

    Args:
        df: DataFrame to add features to
        train_df: Training data to compute statistics from (prevents leakage)

    Returns:
        DataFrame with store features added
    """
    df = df.copy()

    # Use training data if provided, otherwise use the entire dataframe
    # (only appropriate during training with proper CV)
    stats_df = train_df if train_df is not None else df

    # Compute store statistics
    store_stats = stats_df.groupby('Store')[TARGET].agg([
        'mean', 'std', 'min', 'max', 'median'
    ]).reset_index()
    store_stats.columns = ['Store', 'store_mean_sales', 'store_std_sales',
                           'store_min_sales', 'store_max_sales', 'store_median_sales']

    # Store rank by mean sales
    store_stats['store_rank'] = store_stats['store_mean_sales'].rank(ascending=False)
    store_stats['store_percentile'] = store_stats['store_mean_sales'].rank(pct=True)

    # Coefficient of variation
    store_stats['store_cv'] = store_stats['store_std_sales'] / store_stats['store_mean_sales']

    # Size category
    store_stats['store_size'] = pd.cut(
        store_stats['store_mean_sales'],
        bins=3,
        labels=['small', 'medium', 'large']
    )

    # Merge back to main dataframe
    df = df.merge(store_stats, on='Store', how='left')

    return df


def create_economic_features(df: pd.DataFrame, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create engineered features from economic indicators.

    Args:
        df: DataFrame with economic columns
        train_df: Training data for computing baselines

    Returns:
        DataFrame with economic features added
    """
    df = df.copy()
    stats_df = train_df if train_df is not None else df

    # Temperature features
    temp_mean = stats_df['Temperature'].mean()
    temp_std = stats_df['Temperature'].std()
    df['temp_deviation'] = (df['Temperature'] - temp_mean) / temp_std
    df['temp_squared'] = df['Temperature'] ** 2

    # Temperature categories
    df['temp_category'] = pd.cut(
        df['Temperature'],
        bins=[-np.inf, 32, 50, 70, 85, np.inf],
        labels=['freezing', 'cold', 'mild', 'warm', 'hot']
    )

    # Fuel price features
    df['fuel_change'] = df.groupby('Store')['Fuel_Price'].diff()
    df['fuel_pct_change'] = df.groupby('Store')['Fuel_Price'].pct_change()

    # CPI features
    df['cpi_change'] = df.groupby('Store')['CPI'].diff()
    df['cpi_pct_change'] = df.groupby('Store')['CPI'].pct_change()

    # Unemployment features
    df['unemployment_change'] = df.groupby('Store')['Unemployment'].diff()
    unemp_mean = stats_df['Unemployment'].mean()
    df['unemployment_above_avg'] = (df['Unemployment'] > unemp_mean).astype(int)

    # Economic interaction features
    df['cpi_unemployment_ratio'] = df['CPI'] / (df['Unemployment'] + 0.1)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different feature groups.

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()

    # Store x Holiday interaction
    df['store_holiday'] = df['Store'] * df['Holiday_Flag']

    # Store x Month interaction (for seasonal patterns by store)
    df['store_month'] = df['Store'] * df['month']

    # Temperature x Season interaction
    df['temp_quarter'] = df['Temperature'] * df['quarter']

    return df


def engineer_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame = None,
    include_lag: bool = True,
    include_rolling: bool = True
) -> pd.DataFrame:
    """
    Main feature engineering pipeline.

    Args:
        df: Input DataFrame
        train_df: Training data for computing statistics (prevents leakage)
        include_lag: Whether to include lag features
        include_rolling: Whether to include rolling features

    Returns:
        DataFrame with all engineered features
    """
    # Ensure data is sorted
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # Create features in order
    print("Creating temporal features...")
    df = create_temporal_features(df)

    print("Creating cyclical features...")
    df = create_cyclical_features(df)

    print("Creating holiday features...")
    df = create_holiday_features(df)

    if include_lag:
        print("Creating lag features...")
        df = create_lag_features(df)

    if include_rolling:
        print("Creating rolling features...")
        df = create_rolling_features(df)

    print("Creating trend features...")
    df = create_trend_features(df)

    print("Creating store features...")
    df = create_store_features(df, train_df)

    print("Creating economic features...")
    df = create_economic_features(df, train_df)

    print("Creating interaction features...")
    df = create_interaction_features(df)

    print(f"Total features created: {len(df.columns)}")

    return df


def engineer_features_split(
    full_df: pd.DataFrame,
    val_start_date: str,
    include_lag: bool = True,
    include_rolling: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer features on full dataset, then split to prevent lag feature issues.

    This function computes lag/rolling features on the combined data,
    then splits. Store-level features are computed only from training data.

    Args:
        full_df: Complete DataFrame (train + val)
        val_start_date: Date string for validation start (YYYY-MM-DD)
        include_lag: Whether to include lag features
        include_rolling: Whether to include rolling features

    Returns:
        Tuple of (train_featured, val_featured)
    """
    import pandas as pd

    val_start = pd.to_datetime(val_start_date)

    # Sort data
    df = full_df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # Create features that don't need historical data
    print("Creating temporal features...")
    df = create_temporal_features(df)

    print("Creating cyclical features...")
    df = create_cyclical_features(df)

    print("Creating holiday features...")
    df = create_holiday_features(df)

    # Create lag and rolling features on full data (these need history)
    if include_lag:
        print("Creating lag features...")
        df = create_lag_features(df)

    if include_rolling:
        print("Creating rolling features...")
        df = create_rolling_features(df)

    print("Creating trend features...")
    df = create_trend_features(df)

    print("Creating interaction features...")
    df = create_interaction_features(df)

    # Split into train and val
    train_df = df[df['Date'] < val_start].copy()
    val_df = df[df['Date'] >= val_start].copy()

    # Add store features (using only training data to prevent leakage)
    print("Creating store features...")
    train_df = create_store_features(train_df, train_df)
    val_df = create_store_features(val_df, train_df)

    # Add economic features (using training stats)
    print("Creating economic features...")
    train_df = create_economic_features(train_df, train_df)
    val_df = create_economic_features(val_df, train_df)

    print(f"Train features: {train_df.shape}, Val features: {val_df.shape}")

    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns (excluding target, date, and identifiers).

    Args:
        df: DataFrame with features

    Returns:
        List of feature column names
    """
    exclude_cols = ['Date', TARGET, 'temp_category', 'store_size']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Convert categorical columns
    return feature_cols


def handle_missing_features(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in features (especially from lag/rolling).

    Args:
        df: DataFrame with potential missing values
        strategy: 'drop' to remove rows, 'fill' to impute with store mean

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if strategy == 'drop':
        initial_len = len(df)
        df = df.dropna()
        print(f"Dropped {initial_len - len(df)} rows with missing values")
    elif strategy == 'fill':
        # Fill with store mean for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df.groupby('Store')[col].transform(
                    lambda x: x.fillna(x.mean())
                )

    return df


if __name__ == "__main__":
    from .data_preprocessing import load_data, time_based_split

    # Test feature engineering
    df = load_data()
    train_df, val_df = time_based_split(df)

    # Engineer features
    train_featured = engineer_features(train_df)
    print(f"\nFeature columns: {train_featured.shape[1]}")
    print(f"Sample features: {list(train_featured.columns[:20])}")
