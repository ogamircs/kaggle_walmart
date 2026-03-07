"""
Unit tests for data_preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import (
    load_data,
    validate_data,
    time_based_split,
    get_store_data,
    prepare_prophet_data,
    get_data_summary
)


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


class TestLoadData:
    """Tests for load_data function."""

    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a DataFrame."""
        df = load_data()
        assert isinstance(df, pd.DataFrame)

    def test_load_data_has_required_columns(self):
        """Test that loaded data has all required columns."""
        df = load_data()
        required_cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag',
                         'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_load_data_date_column_is_datetime(self):
        """Test that Date column is converted to datetime."""
        df = load_data()
        assert pd.api.types.is_datetime64_any_dtype(df['Date'])

    def test_load_data_sorted_by_store_and_date(self):
        """Test that data is sorted by Store and Date."""
        df = load_data()
        # Check that each store's dates are in ascending order
        for store in df['Store'].unique():
            store_dates = df[df['Store'] == store]['Date'].values
            assert all(store_dates[i] <= store_dates[i+1] for i in range(len(store_dates)-1))

    def test_load_data_no_null_dates(self):
        """Test that there are no null dates."""
        df = load_data()
        assert df['Date'].isnull().sum() == 0


class TestValidateData:
    """Tests for validate_data function."""

    def test_validate_data_returns_dict(self, sample_dataframe):
        """Test that validate_data returns a dictionary."""
        result = validate_data(sample_dataframe)
        assert isinstance(result, dict)

    def test_validate_data_has_required_keys(self, sample_dataframe):
        """Test that validation result has all required keys."""
        result = validate_data(sample_dataframe)
        required_keys = ['total_rows', 'n_stores', 'n_weeks', 'date_range',
                         'missing_values', 'duplicates', 'negative_sales',
                         'complete_coverage']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_validate_data_total_rows(self, sample_dataframe):
        """Test that total_rows matches DataFrame length."""
        result = validate_data(sample_dataframe)
        assert result['total_rows'] == len(sample_dataframe)

    def test_validate_data_n_stores(self, sample_dataframe):
        """Test that n_stores matches unique store count."""
        result = validate_data(sample_dataframe)
        assert result['n_stores'] == sample_dataframe['Store'].nunique()

    def test_validate_data_n_weeks(self, sample_dataframe):
        """Test that n_weeks matches unique date count."""
        result = validate_data(sample_dataframe)
        assert result['n_weeks'] == sample_dataframe['Date'].nunique()

    def test_validate_data_date_range_tuple(self, sample_dataframe):
        """Test that date_range is a tuple with min and max dates."""
        result = validate_data(sample_dataframe)
        assert isinstance(result['date_range'], tuple)
        assert len(result['date_range']) == 2
        assert result['date_range'][0] == sample_dataframe['Date'].min()
        assert result['date_range'][1] == sample_dataframe['Date'].max()

    def test_validate_data_negative_sales_count(self, sample_dataframe):
        """Test negative_sales correctly counts negative values."""
        df = sample_dataframe.copy()
        df.loc[0, 'Weekly_Sales'] = -100
        df.loc[1, 'Weekly_Sales'] = -200
        result = validate_data(df)
        assert result['negative_sales'] == 2


class TestTimeBasedSplit:
    """Tests for time_based_split function."""

    def test_time_based_split_returns_two_dataframes(self, sample_dataframe):
        """Test that time_based_split returns two DataFrames."""
        train_df, val_df = time_based_split(sample_dataframe, '2010-09-01')
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)

    def test_time_based_split_no_overlap(self, sample_dataframe):
        """Test that train and val sets don't overlap in time."""
        split_date = '2010-09-01'
        train_df, val_df = time_based_split(sample_dataframe, split_date)

        split_dt = pd.to_datetime(split_date)
        assert train_df['Date'].max() < split_dt
        assert val_df['Date'].min() >= split_dt

    def test_time_based_split_preserves_all_rows(self, sample_dataframe):
        """Test that all rows are preserved in the split."""
        train_df, val_df = time_based_split(sample_dataframe, '2010-09-01')
        assert len(train_df) + len(val_df) == len(sample_dataframe)

    def test_time_based_split_train_before_val(self, sample_dataframe):
        """Test that all train dates are before all val dates."""
        train_df, val_df = time_based_split(sample_dataframe, '2010-09-01')
        if len(train_df) > 0 and len(val_df) > 0:
            assert train_df['Date'].max() < val_df['Date'].min()


class TestGetStoreData:
    """Tests for get_store_data function."""

    def test_get_store_data_filters_correctly(self, sample_dataframe):
        """Test that get_store_data filters to specific store."""
        store_df = get_store_data(sample_dataframe, 1)
        assert all(store_df['Store'] == 1)

    def test_get_store_data_returns_copy(self, sample_dataframe):
        """Test that get_store_data returns a copy."""
        store_df = get_store_data(sample_dataframe, 1)
        store_df['Weekly_Sales'] = 0
        # Original should not be modified
        assert sample_dataframe['Weekly_Sales'].sum() > 0

    def test_get_store_data_empty_for_nonexistent_store(self, sample_dataframe):
        """Test that non-existent store returns empty DataFrame."""
        store_df = get_store_data(sample_dataframe, 999)
        assert len(store_df) == 0


class TestPrepareProphetData:
    """Tests for prepare_prophet_data function."""

    def test_prepare_prophet_data_column_names(self, sample_dataframe):
        """Test that output has 'ds' and 'y' columns."""
        prophet_df = prepare_prophet_data(sample_dataframe)
        assert 'ds' in prophet_df.columns
        assert 'y' in prophet_df.columns

    def test_prepare_prophet_data_ds_is_date(self, sample_dataframe):
        """Test that 'ds' column contains dates."""
        prophet_df = prepare_prophet_data(sample_dataframe)
        assert pd.api.types.is_datetime64_any_dtype(prophet_df['ds'])

    def test_prepare_prophet_data_y_is_sales(self, sample_dataframe):
        """Test that 'y' column contains sales values."""
        prophet_df = prepare_prophet_data(sample_dataframe)
        assert prophet_df['y'].sum() == sample_dataframe['Weekly_Sales'].sum()

    def test_prepare_prophet_data_with_store_filter(self, sample_dataframe):
        """Test filtering by store_id."""
        prophet_df = prepare_prophet_data(sample_dataframe, store_id=1)
        store_1_sales = sample_dataframe[sample_dataframe['Store'] == 1]['Weekly_Sales'].sum()
        assert prophet_df['y'].sum() == store_1_sales

    def test_prepare_prophet_data_includes_regressors(self, sample_dataframe):
        """Test that regressors are included when available."""
        prophet_df = prepare_prophet_data(sample_dataframe)
        expected_regressors = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
        for reg in expected_regressors:
            assert reg in prophet_df.columns


class TestGetDataSummary:
    """Tests for get_data_summary function."""

    def test_get_data_summary_returns_dict(self, sample_dataframe):
        """Test that get_data_summary returns a dictionary."""
        summary = get_data_summary(sample_dataframe)
        assert isinstance(summary, dict)

    def test_get_data_summary_has_required_keys(self, sample_dataframe):
        """Test that summary has all required keys."""
        summary = get_data_summary(sample_dataframe)
        required_keys = ['shape', 'date_range', 'stores', 'weekly_sales', 'holiday_weeks']
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_get_data_summary_shape(self, sample_dataframe):
        """Test that shape matches DataFrame shape."""
        summary = get_data_summary(sample_dataframe)
        assert summary['shape'] == sample_dataframe.shape

    def test_get_data_summary_weekly_sales_stats(self, sample_dataframe):
        """Test that weekly sales statistics are correct."""
        summary = get_data_summary(sample_dataframe)
        assert abs(summary['weekly_sales']['mean'] - sample_dataframe['Weekly_Sales'].mean()) < 0.01
        assert abs(summary['weekly_sales']['median'] - sample_dataframe['Weekly_Sales'].median()) < 0.01

    def test_get_data_summary_store_count(self, sample_dataframe):
        """Test that store count is correct."""
        summary = get_data_summary(sample_dataframe)
        assert summary['stores']['count'] == sample_dataframe['Store'].nunique()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
