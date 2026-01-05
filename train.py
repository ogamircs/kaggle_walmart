"""
Training pipeline for Walmart Sales Forecasting.
Run this script to train all models and save results.
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_PATH, MODELS_DIR, TARGET, RANDOM_SEED,
    OPTUNA_TRIALS, ORIGINAL_FEATURES
)
from src.data_preprocessing import load_data, time_based_split, validate_data
from src.feature_engineering import (
    engineer_features, engineer_features_split, handle_missing_features, get_feature_columns
)
from src.metrics import (
    calculate_all_metrics, wmape, wmape_per_store,
    format_metrics, compare_models
)
from src.models import (
    RandomForestModel, XGBoostModel, LightGBMModel,
    SARIMAModel, ProphetModel, EnsembleModel,
    train_and_evaluate, get_model
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)


def prepare_ml_data(full_df: pd.DataFrame, val_start_date: str = "2012-09-01"):
    """
    Prepare data for ML models (tree-based).

    Uses engineer_features_split to properly compute lag features on combined
    data before splitting.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names, val_df_with_predictions)
    """
    print("\n" + "="*60)
    print("PREPARING ML DATA")
    print("="*60)

    # Engineer features on full data then split
    print("\nEngineering features (computing lags on full data, then splitting)...")
    train_featured, val_featured = engineer_features_split(full_df, val_start_date)

    # Handle missing values (drop rows with NaN from lag features)
    print("\nHandling missing values...")
    train_clean = handle_missing_features(train_featured, strategy='drop')
    val_clean = handle_missing_features(val_featured, strategy='drop')

    # Get feature columns (exclude target, date, categorical)
    exclude_cols = ['Date', TARGET, 'temp_category', 'store_size']
    feature_cols = [col for col in train_clean.columns
                    if col not in exclude_cols
                    and train_clean[col].dtype in ['int64', 'float64', 'int32', 'float32']]

    print(f"\nUsing {len(feature_cols)} features")

    # Prepare arrays
    X_train = train_clean[feature_cols].values
    y_train = train_clean[TARGET].values
    X_val = val_clean[feature_cols].values
    y_val = val_clean[TARGET].values

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")

    return X_train, y_train, X_val, y_val, feature_cols, val_clean


def train_ml_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune: bool = True
) -> dict:
    """
    Train all ML models (RandomForest, XGBoost, LightGBM).

    Returns:
        Dictionary of {model_name: (model, metrics, predictions)}
    """
    print("\n" + "="*60)
    print("TRAINING ML MODELS")
    print("="*60)

    results = {}

    # RandomForest
    print("\n--- RandomForest ---")
    rf_model = RandomForestModel()
    if tune:
        print("Tuning hyperparameters...")
        best_params = RandomForestModel.tune(X_train, y_train, X_val, y_val, OPTUNA_TRIALS)
        rf_model = RandomForestModel(best_params)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_metrics = calculate_all_metrics(y_val, rf_pred)
    print(f"WMAPE: {rf_metrics['WMAPE']:.2f}%")
    results['RandomForest'] = (rf_model, rf_metrics, rf_pred)

    # XGBoost
    print("\n--- XGBoost ---")
    xgb_model = XGBoostModel()
    if tune:
        print("Tuning hyperparameters...")
        best_params = XGBoostModel.tune(X_train, y_train, X_val, y_val, OPTUNA_TRIALS)
        xgb_model = XGBoostModel(best_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    xgb_metrics = calculate_all_metrics(y_val, xgb_pred)
    print(f"WMAPE: {xgb_metrics['WMAPE']:.2f}%")
    results['XGBoost'] = (xgb_model, xgb_metrics, xgb_pred)

    # LightGBM
    print("\n--- LightGBM ---")
    lgb_model = LightGBMModel()
    if tune:
        print("Tuning hyperparameters...")
        best_params = LightGBMModel.tune(X_train, y_train, X_val, y_val, OPTUNA_TRIALS)
        lgb_model = LightGBMModel(best_params)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    lgb_metrics = calculate_all_metrics(y_val, lgb_pred)
    print(f"WMAPE: {lgb_metrics['WMAPE']:.2f}%")
    results['LightGBM'] = (lgb_model, lgb_metrics, lgb_pred)

    return results


def train_time_series_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> dict:
    """
    Train time series models (SARIMA, Prophet).

    Returns:
        Dictionary of {model_name: (model, metrics, predictions)}
    """
    print("\n" + "="*60)
    print("TRAINING TIME SERIES MODELS")
    print("="*60)

    results = {}
    y_val = val_df[TARGET].values

    # SARIMA
    print("\n--- SARIMA ---")
    try:
        sarima_model = SARIMAModel(order=(1, 0, 1), seasonal_order=(0, 1, 0, 52))
        sarima_model.fit(train_df)

        # For SARIMA, we need to predict step by step
        # Simplified: use last known value + trend
        sarima_pred = []
        for store in val_df['Store'].unique():
            store_train = train_df[train_df['Store'] == store]
            store_val = val_df[val_df['Store'] == store]

            if store in sarima_model.models and sarima_model.models[store] is not None:
                try:
                    forecast = sarima_model.models[store].forecast(steps=len(store_val))
                    sarima_pred.extend(forecast.values)
                except:
                    # Fallback to store mean
                    sarima_pred.extend([store_train[TARGET].mean()] * len(store_val))
            else:
                sarima_pred.extend([store_train[TARGET].mean()] * len(store_val))

        sarima_pred = np.array(sarima_pred)
        sarima_metrics = calculate_all_metrics(y_val, sarima_pred)
        print(f"WMAPE: {sarima_metrics['WMAPE']:.2f}%")
        results['SARIMA'] = (sarima_model, sarima_metrics, sarima_pred)
    except Exception as e:
        print(f"SARIMA training failed: {e}")
        results['SARIMA'] = (None, {'WMAPE': 100}, np.zeros(len(y_val)))

    # Prophet
    print("\n--- Prophet ---")
    try:
        prophet_model = ProphetModel()
        prophet_model.fit(train_df)
        prophet_pred = prophet_model.predict(val_df)
        prophet_metrics = calculate_all_metrics(y_val, prophet_pred)
        print(f"WMAPE: {prophet_metrics['WMAPE']:.2f}%")
        results['Prophet'] = (prophet_model, prophet_metrics, prophet_pred)
    except Exception as e:
        print(f"Prophet training failed: {e}")
        results['Prophet'] = (None, {'WMAPE': 100}, np.zeros(len(y_val)))

    return results


def train_ensemble(
    ml_results: dict,
    ts_results: dict,
    y_val: np.ndarray
) -> tuple:
    """
    Train ensemble model by optimizing weights.

    Returns:
        Tuple of (ensemble_model, metrics, predictions)
    """
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE")
    print("="*60)

    # Collect predictions from all successful models
    predictions = {}
    for name, (model, metrics, pred) in {**ml_results, **ts_results}.items():
        if model is not None and metrics['WMAPE'] < 50:  # Only include reasonable models
            predictions[name] = pred

    if len(predictions) < 2:
        print("Not enough valid models for ensemble")
        return None, {'WMAPE': 100}, np.zeros(len(y_val))

    # Create and optimize ensemble
    ensemble = EnsembleModel()
    ensemble.optimize_weights(predictions, y_val)

    # Make ensemble predictions
    ensemble_pred = ensemble.predict(predictions=predictions)
    ensemble_metrics = calculate_all_metrics(y_val, ensemble_pred)
    print(f"Ensemble WMAPE: {ensemble_metrics['WMAPE']:.2f}%")

    return ensemble, ensemble_metrics, ensemble_pred


def save_results(
    all_results: dict,
    feature_cols: list,
    val_df: pd.DataFrame
):
    """Save models, predictions, and results to disk."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    # Create results directory
    results_dir = MODELS_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    # Save models
    for name, (model, metrics, pred) in all_results.items():
        if model is not None:
            model_path = MODELS_DIR / f"{name.lower()}_model.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")

    # Save feature columns
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")

    # Compile metrics comparison
    metrics_comparison = {
        name: {k: v for k, v in metrics.items()}
        for name, (model, metrics, pred) in all_results.items()
    }

    # Save metrics as JSON
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(metrics_comparison, f, indent=2, default=float)

    # Create predictions dataframe
    predictions_df = val_df[['Store', 'Date', TARGET]].copy()
    for name, (model, metrics, pred) in all_results.items():
        predictions_df[f'{name}_Pred'] = pred

    predictions_df.to_csv(results_dir / "predictions.csv", index=False)
    print(f"Saved predictions to {results_dir / 'predictions.csv'}")

    # Print final comparison
    print("\n" + "="*60)
    print("FINAL MODEL COMPARISON")
    print("="*60)

    comparison_df = pd.DataFrame(metrics_comparison).T
    comparison_df = comparison_df.sort_values('WMAPE')
    print(comparison_df[['WMAPE', 'MAE', 'RMSE', 'R2']].to_string())

    # Best model
    best_model = comparison_df['WMAPE'].idxmin()
    best_wmape = comparison_df.loc[best_model, 'WMAPE']
    print(f"\nBest Model: {best_model} (WMAPE: {best_wmape:.2f}%)")

    return comparison_df


def main(tune: bool = True, skip_time_series: bool = False):
    """
    Main training pipeline.

    Args:
        tune: Whether to tune hyperparameters
        skip_time_series: Whether to skip SARIMA and Prophet (faster)
    """
    print("="*60)
    print("WALMART SALES FORECASTING - TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and validate data
    print("\n--- Loading Data ---")
    df = load_data()
    validation = validate_data(df)
    print(f"Total rows: {validation['total_rows']}")
    print(f"Stores: {validation['n_stores']}")
    print(f"Weeks: {validation['n_weeks']}")

    # Get train/val split for time series models
    print("\n--- Splitting Data ---")
    train_df, val_df = time_based_split(df)

    # Prepare ML data (features engineered on full data, then split)
    X_train, y_train, X_val, y_val, feature_cols, val_clean = prepare_ml_data(df)

    # Train ML models
    ml_results = train_ml_models(X_train, y_train, X_val, y_val, tune=tune)

    # Train time series models
    if not skip_time_series:
        ts_results = train_time_series_models(train_df, val_df)
    else:
        print("\nSkipping time series models...")
        ts_results = {}

    # Train ensemble
    all_results = {**ml_results, **ts_results}
    ensemble_model, ensemble_metrics, ensemble_pred = train_ensemble(
        ml_results, ts_results, y_val
    )
    if ensemble_model is not None:
        all_results['Ensemble'] = (ensemble_model, ensemble_metrics, ensemble_pred)

    # Save results
    comparison_df = save_results(all_results, feature_cols, val_clean)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    return all_results, comparison_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Walmart Sales Forecasting Models")
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--skip-ts', action='store_true', help='Skip time series models')
    args = parser.parse_args()

    main(tune=not args.no_tune, skip_time_series=args.skip_ts)
