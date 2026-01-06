"""
Unit tests for feature_engineering module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import (
    create_temporal_features,
    create_cyclical_features,
    create_holiday_features,
    create_lag_features,
    create_rolling_features,
    create_trend_features,
    create_store_features,
    create_economic_features,
    create_interaction_features,
    engineer_features,
    get_feature_columns,
    handle_missing_features
)
from src.config import TARGET


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range(start='2010-01-01', periods=100, freq='W')
    np.random.seed(42)

    data = {
        'Store': [1] * 50 + [2] * 50,
        'Date': list(dates[:50]) + list(dates[50:]),
        'Weekly_Sales': np.random.uniform(500000, 2000000, 100),
        'Holiday_Flag': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'Temperature': np.random.uniform(30, 90, 100),
        'Fuel_Price': np.random.uniform(2.5, 4.0, 100),
        'CPI': np.random.uniform(180, 230, 100),
        'Unemployment': np.random.uniform(5, 12, 100)
    }

    df = pd.DataFrame(data)
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    return df


class TestCreateTemporalFeatures:
    """Tests for create_temporal_features function."""

    def test_creates_year_column(self, sample_dataframe):
        """Test that year column is created."""
        df = create_temporal_features(sample_dataframe)
        assert 'year' in df.columns
        assert df['year'].iloc[0] == 2010

    def test_creates_month_column(self, sample_dataframe):
        """Test that month column is created."""
        df = create_temporal_features(sample_dataframe)
        assert 'month' in df.columns
        assert df['month'].min() >= 1
        assert df['month'].max() <= 12

    def test_creates_week_of_year_column(self, sample_dataframe):
        """Test that week_of_year column is created."""
        df = create_temporal_features(sample_dataframe)
        assert 'week_of_year' in df.columns
        assert df['week_of_year'].min() >= 1
        assert df['week_of_year'].max() <= 53

    def test_creates_quarter_column(self, sample_dataframe):
        """Test that quarter column is created."""
        df = create_temporal_features(sample_dataframe)
        assert 'quarter' in df.columns
        assert df['quarter'].min() >= 1
        assert df['quarter'].max() <= 4

    def test_creates_boolean_indicators(self, sample_dataframe):
        """Test that boolean indicator columns are created."""
        df = create_temporal_features(sample_dataframe)
        boolean_cols = ['is_month_start', 'is_month_end', 'is_quarter_start',
                        'is_quarter_end', 'is_year_start', 'is_year_end']
        for col in boolean_cols:
            assert col in df.columns
            assert df[col].isin([0, 1]).all()

    def test_returns_copy(self, sample_dataframe):
        """Test that function returns a copy, not modifying original."""
        original_cols = set(sample_dataframe.columns)
        df = create_temporal_features(sample_dataframe)
        assert set(sample_dataframe.columns) == original_cols


class TestCreateCyclicalFeatures:
    """Tests for create_cyclical_features function."""

    def test_creates_month_sin_cos(self, sample_dataframe):
        """Test that month sine/cosine features are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_cyclical_features(df)
        assert 'month_sin' in df.columns
        assert 'month_cos' in df.columns

    def test_creates_week_sin_cos(self, sample_dataframe):
        """Test that week sine/cosine features are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_cyclical_features(df)
        assert 'week_sin' in df.columns
        assert 'week_cos' in df.columns

    def test_sin_cos_range(self, sample_dataframe):
        """Test that sine/cosine values are in [-1, 1]."""
        df = create_temporal_features(sample_dataframe)
        df = create_cyclical_features(df)
        sin_cos_cols = [col for col in df.columns if 'sin' in col or 'cos' in col]
        for col in sin_cos_cols:
            assert df[col].min() >= -1.0
            assert df[col].max() <= 1.0

    def test_sin_cos_are_valid(self, sample_dataframe):
        """Test that sin^2 + cos^2 ≈ 1 for month encoding."""
        df = create_temporal_features(sample_dataframe)
        df = create_cyclical_features(df)
        sum_squares = df['month_sin']**2 + df['month_cos']**2
        assert np.allclose(sum_squares, 1.0, atol=1e-10)


class TestCreateHolidayFeatures:
    """Tests for create_holiday_features function."""

    def test_creates_holiday_indicator_columns(self, sample_dataframe):
        """Test that holiday indicator columns are created."""
        df = create_holiday_features(sample_dataframe)
        holiday_cols = ['is_super_bowl', 'is_labour_day', 'is_thanksgiving', 'is_christmas']
        for col in holiday_cols:
            assert col in df.columns

    def test_creates_pre_holiday_columns(self, sample_dataframe):
        """Test that pre-holiday indicator columns are created."""
        df = create_holiday_features(sample_dataframe)
        assert 'is_pre_thanksgiving' in df.columns
        assert 'is_pre_christmas' in df.columns

    def test_creates_weeks_to_holiday_column(self, sample_dataframe):
        """Test that weeks_to_next_holiday column is created."""
        df = create_holiday_features(sample_dataframe)
        assert 'weeks_to_next_holiday' in df.columns
        assert 'weeks_since_last_holiday' in df.columns

    def test_holiday_indicators_are_binary(self, sample_dataframe):
        """Test that holiday indicators are binary (0 or 1)."""
        df = create_holiday_features(sample_dataframe)
        holiday_cols = [col for col in df.columns if col.startswith('is_')]
        for col in holiday_cols:
            assert df[col].isin([0, 1]).all()


class TestCreateLagFeatures:
    """Tests for create_lag_features function."""

    def test_creates_lag_columns(self, sample_dataframe):
        """Test that lag columns are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_lag_features(df, lags=[1, 2, 4])
        assert 'sales_lag_1' in df.columns
        assert 'sales_lag_2' in df.columns
        assert 'sales_lag_4' in df.columns

    def test_lag_values_are_shifted(self, sample_dataframe):
        """Test that lag values are correctly shifted."""
        df = sample_dataframe.copy()
        df = create_temporal_features(df)
        df = create_lag_features(df, lags=[1])

        # For store 1, check that lag_1 equals previous row's sales
        store_1 = df[df['Store'] == 1].reset_index(drop=True)
        for i in range(1, len(store_1)):
            expected = store_1.loc[i-1, TARGET]
            actual = store_1.loc[i, 'sales_lag_1']
            if not pd.isna(actual):
                assert actual == expected

    def test_first_lag_rows_are_nan(self, sample_dataframe):
        """Test that first rows have NaN for lag features."""
        df = create_temporal_features(sample_dataframe)
        df = create_lag_features(df, lags=[1, 2])

        # First row of each store should have NaN for lag features
        for store in df['Store'].unique():
            store_df = df[df['Store'] == store].iloc[0]
            assert pd.isna(store_df['sales_lag_1'])


class TestCreateRollingFeatures:
    """Tests for create_rolling_features function."""

    def test_creates_rolling_mean_columns(self, sample_dataframe):
        """Test that rolling mean columns are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_rolling_features(df, windows=[4, 8])
        assert 'rolling_mean_4w' in df.columns
        assert 'rolling_mean_8w' in df.columns

    def test_creates_rolling_std_columns(self, sample_dataframe):
        """Test that rolling std columns are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_rolling_features(df, windows=[4, 8])
        assert 'rolling_std_4w' in df.columns
        assert 'rolling_std_8w' in df.columns

    def test_creates_ewma_columns(self, sample_dataframe):
        """Test that EWMA columns are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_rolling_features(df, windows=[4])
        assert 'ewma_4w' in df.columns


class TestCreateTrendFeatures:
    """Tests for create_trend_features function."""

    def test_creates_diff_columns(self, sample_dataframe):
        """Test that difference columns are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_trend_features(df)
        assert 'sales_diff_1' in df.columns
        assert 'sales_diff_4' in df.columns

    def test_creates_pct_change_columns(self, sample_dataframe):
        """Test that percent change columns are created."""
        df = create_temporal_features(sample_dataframe)
        df = create_trend_features(df)
        assert 'sales_pct_change_1' in df.columns
        assert 'sales_pct_change_4' in df.columns


class TestCreateStoreFeatures:
    """Tests for create_store_features function."""

    def test_creates_store_mean_sales(self, sample_dataframe):
        """Test that store mean sales column is created."""
        df = create_store_features(sample_dataframe)
        assert 'store_mean_sales' in df.columns

    def test_creates_store_rank(self, sample_dataframe):
        """Test that store rank column is created."""
        df = create_store_features(sample_dataframe)
        assert 'store_rank' in df.columns

    def test_store_features_same_within_store(self, sample_dataframe):
        """Test that store features are same for all rows of a store."""
        df = create_store_features(sample_dataframe)
        for store in df['Store'].unique():
            store_df = df[df['Store'] == store]
            assert store_df['store_mean_sales'].nunique() == 1

    def test_uses_train_df_when_provided(self, sample_dataframe):
        """Test that train_df is used to prevent data leakage."""
        train_df = sample_dataframe.iloc[:80]
        val_df = sample_dataframe.iloc[80:]

        val_with_features = create_store_features(val_df, train_df)
        train_with_features = create_store_features(train_df, train_df)

        # Store mean from train should be used in val
        for store in val_with_features['Store'].unique():
            train_mean = train_with_features[train_with_features['Store'] == store]['store_mean_sales'].iloc[0]
            val_mean = val_with_features[val_with_features['Store'] == store]['store_mean_sales'].iloc[0]
            assert train_mean == val_mean


class TestCreateEconomicFeatures:
    """Tests for create_economic_features function."""

    def test_creates_temp_deviation(self, sample_dataframe):
        """Test that temperature deviation column is created."""
        df = create_economic_features(sample_dataframe)
        assert 'temp_deviation' in df.columns

    def test_creates_fuel_change(self, sample_dataframe):
        """Test that fuel change column is created."""
        df = create_economic_features(sample_dataframe)
        assert 'fuel_change' in df.columns

    def test_creates_cpi_change(self, sample_dataframe):
        """Test that CPI change column is created."""
        df = create_economic_features(sample_dataframe)
        assert 'cpi_change' in df.columns

    def test_creates_unemployment_features(self, sample_dataframe):
        """Test that unemployment features are created."""
        df = create_economic_features(sample_dataframe)
        assert 'unemployment_change' in df.columns
        assert 'unemployment_above_avg' in df.columns


class TestCreateInteractionFeatures:
    """Tests for create_interaction_features function."""

    def test_creates_store_holiday_interaction(self, sample_dataframe):
        """Test that store-holiday interaction is created."""
        df = create_temporal_features(sample_dataframe)
        df = create_interaction_features(df)
        assert 'store_holiday' in df.columns

    def test_creates_store_month_interaction(self, sample_dataframe):
        """Test that store-month interaction is created."""
        df = create_temporal_features(sample_dataframe)
        df = create_interaction_features(df)
        assert 'store_month' in df.columns

    def test_creates_temp_quarter_interaction(self, sample_dataframe):
        """Test that temp-quarter interaction is created."""
        df = create_temporal_features(sample_dataframe)
        df = create_interaction_features(df)
        assert 'temp_quarter' in df.columns


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_creates_many_features(self, sample_dataframe):
        """Test that many features are created."""
        original_cols = len(sample_dataframe.columns)
        df = engineer_features(sample_dataframe, include_lag=False, include_rolling=False)
        assert len(df.columns) > original_cols

    def test_includes_lag_features_when_enabled(self, sample_dataframe):
        """Test that lag features are included when enabled."""
        df = engineer_features(sample_dataframe, include_lag=True, include_rolling=False)
        lag_cols = [col for col in df.columns if 'lag' in col]
        assert len(lag_cols) > 0

    def test_excludes_lag_features_when_disabled(self, sample_dataframe):
        """Test that lag features are excluded when disabled."""
        df = engineer_features(sample_dataframe, include_lag=False, include_rolling=False)
        lag_cols = [col for col in df.columns if 'sales_lag' in col]
        assert len(lag_cols) == 0

    def test_preserves_original_data(self, sample_dataframe):
        """Test that original data is preserved."""
        df = engineer_features(sample_dataframe, include_lag=False, include_rolling=False)
        assert TARGET in df.columns
        assert 'Store' in df.columns
        assert 'Date' in df.columns


class TestGetFeatureColumns:
    """Tests for get_feature_columns function."""

    def test_excludes_target(self, sample_dataframe):
        """Test that target column is excluded."""
        df = create_temporal_features(sample_dataframe)
        feature_cols = get_feature_columns(df)
        assert TARGET not in feature_cols

    def test_excludes_date(self, sample_dataframe):
        """Test that Date column is excluded."""
        df = create_temporal_features(sample_dataframe)
        feature_cols = get_feature_columns(df)
        assert 'Date' not in feature_cols

    def test_returns_list(self, sample_dataframe):
        """Test that function returns a list."""
        df = create_temporal_features(sample_dataframe)
        feature_cols = get_feature_columns(df)
        assert isinstance(feature_cols, list)


class TestHandleMissingFeatures:
    """Tests for handle_missing_features function."""

    def test_drop_strategy_removes_nan_rows(self, sample_dataframe):
        """Test that 'drop' strategy removes rows with NaN."""
        df = sample_dataframe.copy()
        df.loc[0, 'Temperature'] = np.nan
        df.loc[1, 'Temperature'] = np.nan

        result = handle_missing_features(df, strategy='drop')
        assert len(result) == len(df) - 2

    def test_fill_strategy_fills_nan(self, sample_dataframe):
        """Test that 'fill' strategy fills NaN values."""
        df = sample_dataframe.copy()
        df.loc[0, 'Temperature'] = np.nan

        result = handle_missing_features(df, strategy='fill')
        assert result['Temperature'].isna().sum() == 0

    def test_returns_copy(self, sample_dataframe):
        """Test that function returns a copy."""
        df = sample_dataframe.copy()
        df.loc[0, 'Temperature'] = np.nan

        result = handle_missing_features(df, strategy='drop')
        assert len(sample_dataframe) != len(result) or sample_dataframe['Temperature'].isna().sum() == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
