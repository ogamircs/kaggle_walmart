"""
Unit tests for metrics module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    wmape,
    mape,
    smape,
    calculate_all_metrics,
    wmape_per_store,
    wmape_per_period,
    format_metrics,
    compare_models,
    MetricsTracker
)


@pytest.fixture
def perfect_predictions():
    """Predictions that exactly match actuals."""
    np.random.seed(42)
    y_true = np.random.uniform(1000, 2000, 100)
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def sample_predictions():
    """Sample predictions with some error."""
    np.random.seed(42)
    y_true = np.random.uniform(500000, 2000000, 100)
    # Add ~5% error
    y_pred = y_true * np.random.uniform(0.95, 1.05, 100)
    return y_true, y_pred


@pytest.fixture
def sample_predictions_df():
    """Sample predictions DataFrame."""
    np.random.seed(42)
    n = 100
    y_true = np.random.uniform(500000, 2000000, n)
    y_pred = y_true * np.random.uniform(0.95, 1.05, n)

    return pd.DataFrame({
        'Store': [1] * 50 + [2] * 50,
        'Weekly_Sales': y_true,
        'Predicted': y_pred,
        'Holiday_Flag': np.random.choice([0, 1], n, p=[0.9, 0.1])
    })


class TestWmape:
    """Tests for wmape function."""

    def test_perfect_predictions_zero_error(self, perfect_predictions):
        """Test that perfect predictions give 0% WMAPE."""
        y_true, y_pred = perfect_predictions
        assert wmape(y_true, y_pred) == 0.0

    def test_wmape_is_positive(self, sample_predictions):
        """Test that WMAPE is always positive."""
        y_true, y_pred = sample_predictions
        assert wmape(y_true, y_pred) >= 0

    def test_wmape_is_percentage(self, sample_predictions):
        """Test that WMAPE is a percentage (0-100+ scale)."""
        y_true, y_pred = sample_predictions
        result = wmape(y_true, y_pred)
        # With ~5% error, WMAPE should be around 2.5%
        assert 0 < result < 100

    def test_wmape_length_mismatch_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        with pytest.raises(ValueError):
            wmape(y_true, y_pred)

    def test_wmape_zero_sum_actual_raises_error(self):
        """Test that zero sum of actuals raises ValueError."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            wmape(y_true, y_pred)

    def test_wmape_accepts_lists(self):
        """Test that WMAPE accepts list inputs."""
        y_true = [100, 200, 300]
        y_pred = [100, 200, 300]
        assert wmape(y_true, y_pred) == 0.0

    def test_wmape_known_value(self):
        """Test WMAPE with known values."""
        y_true = np.array([100, 200])
        y_pred = np.array([110, 190])  # 10% error on first, 5% error on second
        # |100-110| + |200-190| = 10 + 10 = 20
        # sum(actual) = 300
        # WMAPE = 20/300 * 100 = 6.67%
        expected = (20 / 300) * 100
        assert abs(wmape(y_true, y_pred) - expected) < 0.01


class TestMape:
    """Tests for mape function."""

    def test_perfect_predictions_zero_error(self, perfect_predictions):
        """Test that perfect predictions give 0% MAPE."""
        y_true, y_pred = perfect_predictions
        assert mape(y_true, y_pred) == 0.0

    def test_mape_is_positive(self, sample_predictions):
        """Test that MAPE is always positive."""
        y_true, y_pred = sample_predictions
        assert mape(y_true, y_pred) >= 0

    def test_mape_excludes_zero_actuals(self):
        """Test that MAPE excludes zero actual values."""
        y_true = np.array([0, 100, 200])
        y_pred = np.array([10, 100, 200])
        # Should only compute on non-zero actuals
        result = mape(y_true, y_pred)
        assert result == 0.0  # 100 and 200 are perfectly predicted

    def test_mape_all_zeros_raises_error(self):
        """Test that all zero actuals raises ValueError."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            mape(y_true, y_pred)

    def test_mape_known_value(self):
        """Test MAPE with known values."""
        y_true = np.array([100, 200])
        y_pred = np.array([110, 220])  # 10% error each
        expected = 10.0
        assert abs(mape(y_true, y_pred) - expected) < 0.01


class TestSmape:
    """Tests for smape function."""

    def test_perfect_predictions_zero_error(self, perfect_predictions):
        """Test that perfect predictions give 0% SMAPE."""
        y_true, y_pred = perfect_predictions
        assert smape(y_true, y_pred) == 0.0

    def test_smape_is_positive(self, sample_predictions):
        """Test that SMAPE is always positive."""
        y_true, y_pred = sample_predictions
        assert smape(y_true, y_pred) >= 0

    def test_smape_symmetric(self):
        """Test that SMAPE is symmetric."""
        y_true = np.array([100, 200])
        y_pred = np.array([150, 250])
        smape1 = smape(y_true, y_pred)
        smape2 = smape(y_pred, y_true)
        assert abs(smape1 - smape2) < 0.01

    def test_smape_handles_zeros(self):
        """Test that SMAPE handles zeros gracefully."""
        y_true = np.array([0, 100])
        y_pred = np.array([0, 100])
        result = smape(y_true, y_pred)
        assert result == 0.0


class TestCalculateAllMetrics:
    """Tests for calculate_all_metrics function."""

    def test_returns_dict(self, sample_predictions):
        """Test that function returns a dictionary."""
        y_true, y_pred = sample_predictions
        result = calculate_all_metrics(y_true, y_pred)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_predictions):
        """Test that result contains all required metric keys."""
        y_true, y_pred = sample_predictions
        result = calculate_all_metrics(y_true, y_pred)
        required_keys = ['WMAPE', 'MAPE', 'SMAPE', 'MAE', 'RMSE', 'R2',
                         'Mean_Error', 'Median_AE']
        for key in required_keys:
            assert key in result

    def test_perfect_predictions(self, perfect_predictions):
        """Test metrics for perfect predictions."""
        y_true, y_pred = perfect_predictions
        result = calculate_all_metrics(y_true, y_pred)
        assert result['WMAPE'] == 0.0
        assert result['MAPE'] == 0.0
        assert result['SMAPE'] == 0.0
        assert result['MAE'] == 0.0
        assert result['RMSE'] == 0.0
        assert result['R2'] == 1.0

    def test_r2_range(self, sample_predictions):
        """Test that R2 is in reasonable range for good predictions."""
        y_true, y_pred = sample_predictions
        result = calculate_all_metrics(y_true, y_pred)
        # With ~5% error, R2 should be close to 1
        assert result['R2'] > 0.9


class TestWmapePerStore:
    """Tests for wmape_per_store function."""

    def test_returns_dataframe(self, sample_predictions_df):
        """Test that function returns a DataFrame."""
        result = wmape_per_store(sample_predictions_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_store_and_wmape_columns(self, sample_predictions_df):
        """Test that result has Store and WMAPE columns."""
        result = wmape_per_store(sample_predictions_df)
        assert 'Store' in result.columns
        assert 'WMAPE' in result.columns

    def test_one_row_per_store(self, sample_predictions_df):
        """Test that there's one row per store."""
        result = wmape_per_store(sample_predictions_df)
        n_stores = sample_predictions_df['Store'].nunique()
        assert len(result) == n_stores


class TestWmapePerPeriod:
    """Tests for wmape_per_period function."""

    def test_returns_dataframe(self, sample_predictions_df):
        """Test that function returns a DataFrame."""
        result = wmape_per_period(sample_predictions_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_period_and_wmape_columns(self, sample_predictions_df):
        """Test that result has period and WMAPE columns."""
        result = wmape_per_period(sample_predictions_df)
        assert 'Holiday_Flag' in result.columns
        assert 'WMAPE' in result.columns

    def test_has_count_column(self, sample_predictions_df):
        """Test that result has Count column."""
        result = wmape_per_period(sample_predictions_df)
        assert 'Count' in result.columns


class TestFormatMetrics:
    """Tests for format_metrics function."""

    def test_returns_string(self, sample_predictions):
        """Test that function returns a string."""
        y_true, y_pred = sample_predictions
        metrics = calculate_all_metrics(y_true, y_pred)
        result = format_metrics(metrics)
        assert isinstance(result, str)

    def test_contains_percentage_format(self, sample_predictions):
        """Test that percentage metrics have % sign."""
        y_true, y_pred = sample_predictions
        metrics = calculate_all_metrics(y_true, y_pred)
        result = format_metrics(metrics)
        assert '%' in result

    def test_contains_dollar_format(self, sample_predictions):
        """Test that monetary metrics have $ sign."""
        y_true, y_pred = sample_predictions
        metrics = calculate_all_metrics(y_true, y_pred)
        result = format_metrics(metrics)
        assert '$' in result


class TestCompareModels:
    """Tests for compare_models function."""

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        results = {
            'Model1': {'WMAPE': 5.0, 'MAE': 1000},
            'Model2': {'WMAPE': 3.0, 'MAE': 800}
        }
        result = compare_models(results)
        assert isinstance(result, pd.DataFrame)

    def test_sorted_by_primary_metric(self):
        """Test that result is sorted by primary metric."""
        results = {
            'Model1': {'WMAPE': 5.0, 'MAE': 1000},
            'Model2': {'WMAPE': 3.0, 'MAE': 800},
            'Model3': {'WMAPE': 4.0, 'MAE': 900}
        }
        result = compare_models(results, primary_metric='WMAPE')
        assert result.iloc[0]['WMAPE'] == 3.0  # Lowest WMAPE first

    def test_has_rank_column(self):
        """Test that result has Rank column."""
        results = {
            'Model1': {'WMAPE': 5.0},
            'Model2': {'WMAPE': 3.0}
        }
        result = compare_models(results)
        assert 'Rank' in result.columns


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_init_empty_history(self):
        """Test that tracker initializes with empty history."""
        tracker = MetricsTracker()
        assert len(tracker.history) == 0

    def test_log_adds_entry(self):
        """Test that log adds an entry to history."""
        tracker = MetricsTracker()
        tracker.log(1, {'WMAPE': 10.0})
        assert len(tracker.history) == 1

    def test_log_with_val_metrics(self):
        """Test logging with both train and val metrics."""
        tracker = MetricsTracker()
        tracker.log(1, {'WMAPE': 10.0}, {'WMAPE': 12.0})
        entry = tracker.history[0]
        assert 'train_WMAPE' in entry
        assert 'val_WMAPE' in entry

    def test_get_history_returns_dataframe(self):
        """Test that get_history returns a DataFrame."""
        tracker = MetricsTracker()
        tracker.log(1, {'WMAPE': 10.0})
        result = tracker.get_history()
        assert isinstance(result, pd.DataFrame)

    def test_best_epoch(self):
        """Test that best_epoch returns correct epoch."""
        tracker = MetricsTracker()
        tracker.log(0, {'WMAPE': 10.0}, {'WMAPE': 15.0})
        tracker.log(1, {'WMAPE': 8.0}, {'WMAPE': 12.0})
        tracker.log(2, {'WMAPE': 7.0}, {'WMAPE': 14.0})
        # Best val_WMAPE is at epoch 1
        assert tracker.best_epoch('val_WMAPE') == 1

    def test_best_epoch_missing_metric(self):
        """Test best_epoch with missing metric returns -1."""
        tracker = MetricsTracker()
        tracker.log(0, {'WMAPE': 10.0})
        assert tracker.best_epoch('nonexistent') == -1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
