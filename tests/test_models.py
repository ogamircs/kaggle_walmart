"""
Unit tests for models module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BaseModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    EnsembleModel,
    get_model,
    train_and_evaluate
)
from src.config import RANDOM_SEED


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(RANDOM_SEED)
    n_samples = 200

    X = np.random.randn(n_samples, 10)
    # Create a simple linear relationship with noise
    y = X[:, 0] * 100000 + X[:, 1] * 50000 + np.random.randn(n_samples) * 10000 + 1000000

    return X, y


@pytest.fixture
def train_val_split(sample_data):
    """Split sample data into train and validation sets."""
    X, y = sample_data
    split_idx = int(len(X) * 0.8)

    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_val, y_val


class TestRandomForestModel:
    """Tests for RandomForestModel class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = RandomForestModel()
        assert model.name == "RandomForest"
        assert model.is_fitted == False
        assert model.model is None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {'n_estimators': 50, 'max_depth': 5}
        model = RandomForestModel(params)
        assert model.params == params

    def test_fit(self, sample_data):
        """Test fitting the model."""
        X, y = sample_data
        model = RandomForestModel({'n_estimators': 10})
        model.fit(X, y)

        assert model.is_fitted == True
        assert model.model is not None
        assert model.feature_importance_ is not None

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict before fit raises ValueError."""
        X, y = sample_data
        model = RandomForestModel()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_after_fit(self, train_val_split):
        """Test predictions after fitting."""
        X_train, y_train, X_val, y_val = train_val_split
        model = RandomForestModel({'n_estimators': 10})
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_val)

    def test_feature_importance_shape(self, sample_data):
        """Test that feature importance has correct shape."""
        X, y = sample_data
        model = RandomForestModel({'n_estimators': 10})
        model.fit(X, y)

        assert len(model.feature_importance_) == X.shape[1]

    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self for chaining."""
        X, y = sample_data
        model = RandomForestModel({'n_estimators': 10})
        result = model.fit(X, y)

        assert result is model


class TestXGBoostModel:
    """Tests for XGBoostModel class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = XGBoostModel()
        assert model.name == "XGBoost"
        assert model.is_fitted == False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {'n_estimators': 50, 'max_depth': 3}
        model = XGBoostModel(params)
        assert model.params == params

    def test_fit(self, sample_data):
        """Test fitting the model."""
        X, y = sample_data
        model = XGBoostModel({'n_estimators': 10})
        model.fit(X, y)

        assert model.is_fitted == True
        assert model.model is not None

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict before fit raises ValueError."""
        X, y = sample_data
        model = XGBoostModel()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_after_fit(self, train_val_split):
        """Test predictions after fitting."""
        X_train, y_train, X_val, y_val = train_val_split
        model = XGBoostModel({'n_estimators': 10})
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_val)


class TestLightGBMModel:
    """Tests for LightGBMModel class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = LightGBMModel()
        assert model.name == "LightGBM"
        assert model.is_fitted == False

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {'n_estimators': 50, 'max_depth': 3}
        model = LightGBMModel(params)
        assert model.params == params

    def test_fit(self, sample_data):
        """Test fitting the model."""
        X, y = sample_data
        model = LightGBMModel({'n_estimators': 10, 'verbose': -1})
        model.fit(X, y)

        assert model.is_fitted == True
        assert model.model is not None

    def test_predict_before_fit_raises_error(self, sample_data):
        """Test that predict before fit raises ValueError."""
        X, y = sample_data
        model = LightGBMModel()

        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict(X)

    def test_predict_after_fit(self, train_val_split):
        """Test predictions after fitting."""
        X_train, y_train, X_val, y_val = train_val_split
        model = LightGBMModel({'n_estimators': 10, 'verbose': -1})
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_val)


class TestEnsembleModel:
    """Tests for EnsembleModel class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        model = EnsembleModel()
        assert model.name == "Ensemble"
        assert model.base_models == {}
        assert model.weights is None

    def test_init_with_models(self):
        """Test initialization with base models."""
        rf = RandomForestModel()
        xgb = XGBoostModel()
        model = EnsembleModel({'RF': rf, 'XGB': xgb})

        assert 'RF' in model.base_models
        assert 'XGB' in model.base_models

    def test_add_model(self):
        """Test adding a model to ensemble."""
        model = EnsembleModel()
        rf = RandomForestModel()
        model.add_model('RF', rf)

        assert 'RF' in model.base_models

    def test_fit_sets_is_fitted(self):
        """Test that fit sets is_fitted flag."""
        model = EnsembleModel()
        model.fit()

        assert model.is_fitted == True

    def test_optimize_weights(self):
        """Test weight optimization."""
        model = EnsembleModel()

        predictions = {
            'Model1': np.array([100, 200, 300]),
            'Model2': np.array([110, 190, 310])
        }
        y_true = np.array([105, 195, 305])

        weights = model.optimize_weights(predictions, y_true)

        assert 'Model1' in weights
        assert 'Model2' in weights
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Weights sum to 1

    def test_predict_with_equal_weights(self):
        """Test prediction with default equal weights."""
        model = EnsembleModel()
        model.is_fitted = True

        predictions = {
            'Model1': np.array([100, 200]),
            'Model2': np.array([200, 300])
        }

        result = model.predict(predictions=predictions)

        # With equal weights: (100+200)/2=150, (200+300)/2=250
        expected = np.array([150, 250])
        np.testing.assert_array_almost_equal(result, expected)

    def test_predict_with_custom_weights(self):
        """Test prediction with custom weights."""
        model = EnsembleModel()
        model.is_fitted = True
        model.weights = {'Model1': 0.75, 'Model2': 0.25}

        predictions = {
            'Model1': np.array([100, 200]),
            'Model2': np.array([200, 300])
        }

        result = model.predict(predictions=predictions)

        # 0.75*100 + 0.25*200 = 125, 0.75*200 + 0.25*300 = 225
        expected = np.array([125, 225])
        np.testing.assert_array_almost_equal(result, expected)


class TestGetModel:
    """Tests for get_model factory function."""

    def test_get_random_forest(self):
        """Test getting RandomForest model."""
        model = get_model('RandomForest')
        assert isinstance(model, RandomForestModel)

    def test_get_xgboost(self):
        """Test getting XGBoost model."""
        model = get_model('XGBoost')
        assert isinstance(model, XGBoostModel)

    def test_get_lightgbm(self):
        """Test getting LightGBM model."""
        model = get_model('LightGBM')
        assert isinstance(model, LightGBMModel)

    def test_get_ensemble(self):
        """Test getting Ensemble model."""
        model = get_model('Ensemble')
        assert isinstance(model, EnsembleModel)

    def test_get_with_params(self):
        """Test getting model with parameters."""
        params = {'n_estimators': 50}
        model = get_model('RandomForest', params)
        assert model.params == params

    def test_get_unknown_model_raises_error(self):
        """Test that unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model('UnknownModel')


class TestTrainAndEvaluate:
    """Tests for train_and_evaluate function."""

    def test_returns_model_and_metrics(self, train_val_split):
        """Test that function returns model and metrics."""
        X_train, y_train, X_val, y_val = train_val_split
        model = RandomForestModel({'n_estimators': 10})

        fitted_model, metrics = train_and_evaluate(
            model, X_train, y_train, X_val, y_val, tune=False
        )

        assert fitted_model.is_fitted == True
        assert isinstance(metrics, dict)
        assert 'WMAPE' in metrics

    def test_model_is_fitted(self, train_val_split):
        """Test that returned model is fitted."""
        X_train, y_train, X_val, y_val = train_val_split
        model = RandomForestModel({'n_estimators': 10})

        fitted_model, _ = train_and_evaluate(
            model, X_train, y_train, X_val, y_val, tune=False
        )

        # Should be able to predict
        predictions = fitted_model.predict(X_val)
        assert len(predictions) == len(X_val)


class TestModelSaveLoad:
    """Tests for model save/load functionality."""

    def test_save_creates_file(self, sample_data, tmp_path):
        """Test that save creates a file."""
        X, y = sample_data
        model = RandomForestModel({'n_estimators': 10})
        model.fit(X, y)

        save_path = tmp_path / "test_model.joblib"
        model.save(save_path)

        assert save_path.exists()

    def test_load_recovers_model(self, sample_data, tmp_path):
        """Test that load recovers the saved model."""
        X, y = sample_data
        model = RandomForestModel({'n_estimators': 10})
        model.fit(X, y)

        save_path = tmp_path / "test_model.joblib"
        model.save(save_path)

        loaded_model = RandomForestModel.load(save_path)

        assert loaded_model.is_fitted == True
        assert loaded_model.name == "RandomForest"

    def test_loaded_model_can_predict(self, train_val_split, tmp_path):
        """Test that loaded model can make predictions."""
        X_train, y_train, X_val, y_val = train_val_split
        model = RandomForestModel({'n_estimators': 10})
        model.fit(X_train, y_train)

        original_predictions = model.predict(X_val)

        save_path = tmp_path / "test_model.joblib"
        model.save(save_path)
        loaded_model = RandomForestModel.load(save_path)

        loaded_predictions = loaded_model.predict(X_val)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
