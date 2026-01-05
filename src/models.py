"""
Model implementations for Walmart Sales Forecasting.
Includes: RandomForest, XGBoost, LightGBM, SARIMA, Prophet, and Ensemble.
"""
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

import optuna
from tqdm import tqdm

from .config import (
    RANDOM_SEED, MODELS_DIR, OPTUNA_TRIALS, HOLIDAYS,
    RF_PARAM_SPACE, XGB_PARAM_SPACE, LGB_PARAM_SPACE
)
from .metrics import wmape, calculate_all_metrics

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def save(self, path: Path = None):
        """Save the model to disk."""
        if path is None:
            path = MODELS_DIR / f"{self.name}.joblib"
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load a model from disk."""
        return joblib.load(path)


class RandomForestModel(BaseModel):
    """Random Forest Regressor wrapper with Optuna tuning."""

    def __init__(self, params: Dict = None):
        super().__init__("RandomForest")
        self.params = params or {}
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'n_jobs': -1,
            'random_state': RANDOM_SEED
        }

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit Random Forest model."""
        final_params = {**self.default_params, **self.params}
        self.model = RandomForestRegressor(**final_params)
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)

    @staticmethod
    def tune(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             n_trials: int = OPTUNA_TRIALS) -> Dict:
        """Tune hyperparameters using Optuna."""

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *RF_PARAM_SPACE['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *RF_PARAM_SPACE['max_depth']),
                'min_samples_split': trial.suggest_int('min_samples_split', *RF_PARAM_SPACE['min_samples_split']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', *RF_PARAM_SPACE['min_samples_leaf']),
                'n_jobs': -1,
                'random_state': RANDOM_SEED
            }

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return wmape(y_val, y_pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params


class XGBoostModel(BaseModel):
    """XGBoost Regressor wrapper with Optuna tuning."""

    def __init__(self, params: Dict = None):
        super().__init__("XGBoost")
        self.params = params or {}
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': RANDOM_SEED,
            'n_jobs': -1
        }

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit XGBoost model."""
        import xgboost as xgb

        final_params = {**self.default_params, **self.params}
        self.model = xgb.XGBRegressor(**final_params)
        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)

    @staticmethod
    def tune(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             n_trials: int = OPTUNA_TRIALS) -> Dict:
        """Tune hyperparameters using Optuna."""
        import xgboost as xgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *XGB_PARAM_SPACE['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *XGB_PARAM_SPACE['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *XGB_PARAM_SPACE['learning_rate'], log=True),
                'subsample': trial.suggest_float('subsample', *XGB_PARAM_SPACE['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *XGB_PARAM_SPACE['colsample_bytree']),
                'min_child_weight': trial.suggest_int('min_child_weight', *XGB_PARAM_SPACE['min_child_weight']),
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            }

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_val)
            return wmape(y_val, y_pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params


class LightGBMModel(BaseModel):
    """LightGBM Regressor wrapper with Optuna tuning."""

    def __init__(self, params: Dict = None):
        super().__init__("LightGBM")
        self.params = params or {}
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1
        }

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit LightGBM model."""
        import lightgbm as lgb

        final_params = {**self.default_params, **self.params}
        self.model = lgb.LGBMRegressor(**final_params)
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)

    @staticmethod
    def tune(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             n_trials: int = OPTUNA_TRIALS) -> Dict:
        """Tune hyperparameters using Optuna."""
        import lightgbm as lgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', *LGB_PARAM_SPACE['n_estimators']),
                'max_depth': trial.suggest_int('max_depth', *LGB_PARAM_SPACE['max_depth']),
                'learning_rate': trial.suggest_float('learning_rate', *LGB_PARAM_SPACE['learning_rate'], log=True),
                'num_leaves': trial.suggest_int('num_leaves', *LGB_PARAM_SPACE['num_leaves']),
                'subsample': trial.suggest_float('subsample', *LGB_PARAM_SPACE['subsample']),
                'colsample_bytree': trial.suggest_float('colsample_bytree', *LGB_PARAM_SPACE['colsample_bytree']),
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return wmape(y_val, y_pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return study.best_params


class SARIMAModel(BaseModel):
    """SARIMA model trained per store."""

    def __init__(self, order: Tuple = (1, 1, 1), seasonal_order: Tuple = (1, 1, 0, 52)):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}  # Store ID -> fitted model

    def fit(self, df: pd.DataFrame, **kwargs):
        """
        Fit SARIMA model for each store.

        Args:
            df: DataFrame with Store, Date, and Weekly_Sales columns
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        stores = df['Store'].unique()
        print(f"Fitting SARIMA for {len(stores)} stores...")

        for store in tqdm(stores, desc="SARIMA fitting"):
            store_data = df[df['Store'] == store].set_index('Date')['Weekly_Sales']

            try:
                model = SARIMAX(
                    store_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.models[store] = model.fit(disp=False, maxiter=100)
            except Exception as e:
                print(f"Warning: SARIMA failed for store {store}: {e}")
                self.models[store] = None

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for validation data.

        Args:
            df: DataFrame with Store and Date columns

        Returns:
            Array of predictions
        """
        predictions = []

        for idx, row in df.iterrows():
            store = row['Store']
            if store in self.models and self.models[store] is not None:
                try:
                    # Forecast next step
                    pred = self.models[store].forecast(steps=1).values[0]
                    predictions.append(pred)
                except:
                    # Fallback to store mean
                    predictions.append(df[df['Store'] == store]['Weekly_Sales'].mean()
                                       if 'Weekly_Sales' in df.columns else 1000000)
            else:
                predictions.append(df[df['Store'] == store]['Weekly_Sales'].mean()
                                   if 'Weekly_Sales' in df.columns else 1000000)

        return np.array(predictions)

    def predict_store(self, store_id: int, steps: int = 1) -> np.ndarray:
        """Predict for a specific store."""
        if store_id not in self.models or self.models[store_id] is None:
            raise ValueError(f"No fitted model for store {store_id}")
        return self.models[store_id].forecast(steps=steps).values


class ProphetModel(BaseModel):
    """Facebook Prophet model trained per store."""

    def __init__(self):
        super().__init__("Prophet")
        self.models = {}  # Store ID -> fitted model

    def fit(self, df: pd.DataFrame, **kwargs):
        """
        Fit Prophet model for each store.

        Args:
            df: DataFrame with Store, Date, Weekly_Sales, and regressor columns
        """
        from prophet import Prophet

        # Create holiday dataframe for Prophet
        holiday_df = self._create_holiday_df()

        stores = df['Store'].unique()
        print(f"Fitting Prophet for {len(stores)} stores...")

        for store in tqdm(stores, desc="Prophet fitting"):
            store_data = df[df['Store'] == store][['Date', 'Weekly_Sales', 'Temperature',
                                                     'Fuel_Price', 'CPI', 'Unemployment']].copy()
            store_data.columns = ['ds', 'y', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

            try:
                model = Prophet(
                    holidays=holiday_df,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    holidays_prior_scale=10.0,
                    seasonality_mode='multiplicative'
                )

                # Add regressors
                model.add_regressor('Temperature')
                model.add_regressor('Fuel_Price')
                model.add_regressor('CPI')
                model.add_regressor('Unemployment')

                self.models[store] = model.fit(store_data)
            except Exception as e:
                print(f"Warning: Prophet failed for store {store}: {e}")
                self.models[store] = None

        self.is_fitted = True
        return self

    def _create_holiday_df(self) -> pd.DataFrame:
        """Create holiday dataframe for Prophet."""
        holidays_list = []
        for holiday_name, dates in HOLIDAYS.items():
            for date in dates:
                holidays_list.append({
                    'holiday': holiday_name.lower(),
                    'ds': date,
                    'lower_window': -1,
                    'upper_window': 1
                })
        return pd.DataFrame(holidays_list)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for validation data.

        Args:
            df: DataFrame with Store, Date, and regressor columns

        Returns:
            Array of predictions
        """
        predictions = []

        for store in df['Store'].unique():
            store_mask = df['Store'] == store
            store_data = df[store_mask][['Date', 'Temperature', 'Fuel_Price',
                                          'CPI', 'Unemployment']].copy()
            store_data.columns = ['ds', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

            if store in self.models and self.models[store] is not None:
                try:
                    forecast = self.models[store].predict(store_data)
                    store_preds = forecast['yhat'].values
                except Exception as e:
                    # Fallback
                    store_preds = np.full(len(store_data), 1000000)
            else:
                store_preds = np.full(len(store_data), 1000000)

            predictions.extend(store_preds)

        return np.array(predictions)


class EnsembleModel(BaseModel):
    """Ensemble model combining predictions from multiple models."""

    def __init__(self, models: Dict[str, BaseModel] = None):
        super().__init__("Ensemble")
        self.base_models = models or {}
        self.weights = None

    def add_model(self, name: str, model: BaseModel):
        """Add a model to the ensemble."""
        self.base_models[name] = model

    def fit(self, X: np.ndarray = None, y: np.ndarray = None, **kwargs):
        """
        Fit ensemble by optimizing weights on validation predictions.

        Note: Base models should already be fitted.
        """
        self.is_fitted = True
        return self

    def optimize_weights(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Optimize ensemble weights to minimize WMAPE.

        Args:
            predictions: Dict of model_name -> predictions array
            y_true: True values
        """
        from scipy.optimize import minimize

        model_names = list(predictions.keys())
        n_models = len(model_names)

        def objective(weights):
            weights = weights / weights.sum()  # Normalize
            ensemble_pred = sum(
                w * predictions[name]
                for w, name in zip(weights, model_names)
            )
            return wmape(y_true, ensemble_pred)

        # Initial equal weights
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=[(0, 1)] * n_models,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
        )

        self.weights = dict(zip(model_names, result.x))
        print(f"Optimized weights: {self.weights}")

        return self.weights

    def predict(self, X: np.ndarray = None, predictions: Dict[str, np.ndarray] = None) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Features (for base models)
            predictions: Pre-computed predictions from base models

        Returns:
            Ensemble predictions
        """
        if predictions is None:
            # Get predictions from base models
            predictions = {
                name: model.predict(X)
                for name, model in self.base_models.items()
            }

        if self.weights is None:
            # Equal weights if not optimized
            n_models = len(predictions)
            self.weights = {name: 1/n_models for name in predictions.keys()}

        # Weighted average
        ensemble_pred = sum(
            self.weights.get(name, 0) * preds
            for name, preds in predictions.items()
        )

        return ensemble_pred


def get_model(model_name: str, params: Dict = None) -> BaseModel:
    """
    Factory function to get a model by name.

    Args:
        model_name: One of 'RandomForest', 'XGBoost', 'LightGBM', 'SARIMA', 'Prophet', 'Ensemble'
        params: Optional parameters for the model

    Returns:
        Model instance
    """
    models = {
        'RandomForest': RandomForestModel,
        'XGBoost': XGBoostModel,
        'LightGBM': LightGBMModel,
        'SARIMA': SARIMAModel,
        'Prophet': ProphetModel,
        'Ensemble': EnsembleModel
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    if params:
        return models[model_name](params)
    return models[model_name]()


def train_and_evaluate(
    model: BaseModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune: bool = False
) -> Tuple[BaseModel, Dict]:
    """
    Train a model and evaluate on validation set.

    Args:
        model: Model instance
        X_train, y_train: Training data
        X_val, y_val: Validation data
        tune: Whether to tune hyperparameters

    Returns:
        Tuple of (fitted model, metrics dict)
    """
    if tune and hasattr(model, 'tune'):
        print(f"Tuning {model.name}...")
        best_params = model.__class__.tune(X_train, y_train, X_val, y_val)
        model.params = best_params
        print(f"Best params: {best_params}")

    print(f"Training {model.name}...")
    model.fit(X_train, y_train)

    print(f"Evaluating {model.name}...")
    y_pred = model.predict(X_val)
    metrics = calculate_all_metrics(y_val, y_pred)

    print(f"{model.name} WMAPE: {metrics['WMAPE']:.2f}%")

    return model, metrics
