"""
Unit tests for gradio_app module.
Tests the plotting and prediction functions (not the Gradio UI itself).
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_data():
    """Create mock global data for testing."""
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


@pytest.fixture
def mock_predictions_df(mock_data):
    """Create mock predictions DataFrame."""
    df = mock_data.copy()
    df['RandomForest_Pred'] = df['Weekly_Sales'] * np.random.uniform(0.95, 1.05, len(df))
    df['XGBoost_Pred'] = df['Weekly_Sales'] * np.random.uniform(0.95, 1.05, len(df))
    return df


@pytest.fixture
def mock_metrics():
    """Create mock metrics dictionary."""
    return {
        'RandomForest': {'WMAPE': 2.5, 'MAE': 50000, 'RMSE': 75000, 'R2': 0.95},
        'XGBoost': {'WMAPE': 2.3, 'MAE': 48000, 'RMSE': 72000, 'R2': 0.96}
    }


class TestPlotFunctions:
    """Tests for plotting functions."""

    def test_plot_sales_distribution(self, mock_data):
        """Test plot_sales_distribution returns a figure."""
        # Import and patch the global DATA variable
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_sales_distribution()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_sales_over_time_all_stores(self, mock_data):
        """Test plot_sales_over_time with all stores."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_sales_over_time(None)

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_sales_over_time_single_store(self, mock_data):
        """Test plot_sales_over_time with single store."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_sales_over_time("1")

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_sales_by_store(self, mock_data):
        """Test plot_sales_by_store returns a figure."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_sales_by_store()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_holiday_impact(self, mock_data):
        """Test plot_holiday_impact returns a figure."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_holiday_impact()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_seasonality(self, mock_data):
        """Test plot_seasonality returns a figure."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_seasonality()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_correlation_matrix(self, mock_data):
        """Test plot_correlation_matrix returns a figure."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_correlation_matrix()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_store_heatmap(self, mock_data):
        """Test plot_store_heatmap returns a figure."""
        from app import gradio_app
        gradio_app.DATA = mock_data

        fig = gradio_app.plot_store_heatmap()

        assert fig is not None
        assert isinstance(fig, go.Figure)


class TestResultsFunctions:
    """Tests for results visualization functions."""

    def test_get_metrics_table_with_metrics(self, mock_metrics):
        """Test get_metrics_table returns DataFrame when metrics exist."""
        from app import gradio_app
        gradio_app.METRICS = mock_metrics

        result = gradio_app.get_metrics_table()

        assert isinstance(result, pd.DataFrame)
        assert 'Model' in result.columns

    def test_get_metrics_table_empty(self):
        """Test get_metrics_table returns message when no metrics."""
        from app import gradio_app
        gradio_app.METRICS = {}

        result = gradio_app.get_metrics_table()

        assert isinstance(result, pd.DataFrame)
        assert 'Message' in result.columns

    def test_plot_actual_vs_predicted_with_data(self, mock_predictions_df):
        """Test plot_actual_vs_predicted with predictions."""
        from app import gradio_app
        gradio_app.PREDICTIONS_DF = mock_predictions_df

        fig = gradio_app.plot_actual_vs_predicted()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_actual_vs_predicted_no_data(self):
        """Test plot_actual_vs_predicted without predictions."""
        from app import gradio_app
        gradio_app.PREDICTIONS_DF = None

        fig = gradio_app.plot_actual_vs_predicted()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_wmape_by_store_with_data(self, mock_predictions_df):
        """Test plot_wmape_by_store with predictions."""
        from app import gradio_app
        gradio_app.PREDICTIONS_DF = mock_predictions_df

        fig = gradio_app.plot_wmape_by_store()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_residuals_with_data(self, mock_predictions_df):
        """Test plot_residuals with predictions."""
        from app import gradio_app
        gradio_app.PREDICTIONS_DF = mock_predictions_df

        fig = gradio_app.plot_residuals()

        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_plot_residuals_no_data(self):
        """Test plot_residuals without predictions."""
        from app import gradio_app
        gradio_app.PREDICTIONS_DF = None

        fig = gradio_app.plot_residuals()

        assert fig is not None
        assert isinstance(fig, go.Figure)


class TestPredictionFunction:
    """Tests for make_prediction function."""

    def test_make_prediction_no_model(self, mock_data):
        """Test make_prediction without a model returns error message."""
        from app import gradio_app
        gradio_app.SIMPLE_MODEL = None
        gradio_app.DATA = mock_data

        result = gradio_app.make_prediction(
            store="1",
            date_str="15-10-2012",
            temperature=60,
            fuel_price=3.5,
            cpi=210,
            unemployment=8.0,
            is_holiday=False
        )

        assert "not available" in result.lower()

    def test_make_prediction_invalid_date(self, mock_data):
        """Test make_prediction with invalid date returns error."""
        from app import gradio_app
        gradio_app.SIMPLE_MODEL = MagicMock()
        gradio_app.SIMPLE_FEATURE_COLS = ['Store', 'Temperature']
        gradio_app.DATA = mock_data

        result = gradio_app.make_prediction(
            store="1",
            date_str="invalid-date",
            temperature=60,
            fuel_price=3.5,
            cpi=210,
            unemployment=8.0,
            is_holiday=False
        )

        assert "invalid date" in result.lower()

    def test_make_prediction_with_model(self, mock_data):
        """Test make_prediction with a model returns prediction."""
        from app import gradio_app

        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1000000])

        gradio_app.SIMPLE_MODEL = mock_model
        gradio_app.SIMPLE_FEATURE_COLS = [
            'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'year', 'month', 'week_of_year', 'quarter',
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'temp_squared', 'cpi_unemployment'
        ]
        gradio_app.DATA = mock_data

        result = gradio_app.make_prediction(
            store="1",
            date_str="15-10-2012",
            temperature=60,
            fuel_price=3.5,
            cpi=210,
            unemployment=8.0,
            is_holiday=False
        )

        assert "$" in result  # Should contain formatted price
        assert "1,000,000" in result  # Should contain the prediction

    def test_make_prediction_holiday_flag(self, mock_data):
        """Test make_prediction correctly handles holiday flag."""
        from app import gradio_app

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1500000])

        gradio_app.SIMPLE_MODEL = mock_model
        gradio_app.SIMPLE_FEATURE_COLS = [
            'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'year', 'month', 'week_of_year', 'quarter',
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            'temp_squared', 'cpi_unemployment'
        ]
        gradio_app.DATA = mock_data

        result = gradio_app.make_prediction(
            store="1",
            date_str="15-10-2012",
            temperature=60,
            fuel_price=3.5,
            cpi=210,
            unemployment=8.0,
            is_holiday=True
        )

        assert "Yes" in result  # Holiday should show "Yes"


class TestDataLoading:
    """Tests for data loading functions."""

    @patch('app.gradio_app.load_data')
    @patch('app.gradio_app.time_based_split')
    def test_load_all_data_sets_global_variables(self, mock_split, mock_load, mock_data):
        """Test that load_all_data sets global variables."""
        from app import gradio_app

        mock_load.return_value = mock_data
        mock_split.return_value = (mock_data.iloc[:80], mock_data.iloc[80:])

        # Mock Path.glob to return empty list (no model files)
        with patch.object(Path, 'glob', return_value=[]):
            with patch.object(Path, 'exists', return_value=False):
                gradio_app.load_all_data()

        assert gradio_app.DATA is not None
        assert gradio_app.TRAIN_DF is not None
        assert gradio_app.VAL_DF is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
