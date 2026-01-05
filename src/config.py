"""
Configuration and constants for Walmart Sales Forecasting.
"""
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "Walmart.csv"
MODELS_DIR = BASE_DIR / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Validation split date (last 2 months)
VALIDATION_START_DATE = "2012-09-01"

# Holiday dates
HOLIDAYS = {
    'Super_Bowl': pd.to_datetime(['2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08']),
    'Labour_Day': pd.to_datetime(['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06']),
    'Thanksgiving': pd.to_datetime(['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']),
    'Christmas': pd.to_datetime(['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'])
}

# All holiday dates combined for easy lookup
ALL_HOLIDAY_DATES = pd.DatetimeIndex(
    list(HOLIDAYS['Super_Bowl']) +
    list(HOLIDAYS['Labour_Day']) +
    list(HOLIDAYS['Thanksgiving']) +
    list(HOLIDAYS['Christmas'])
)

# Feature groups for selection
FEATURE_GROUPS = {
    'temporal': ['year', 'month', 'week_of_year', 'quarter', 'day_of_year',
                 'is_month_start', 'is_month_end', 'week_of_month'],
    'cyclical': ['month_sin', 'month_cos', 'week_sin', 'week_cos'],
    'holiday': ['is_super_bowl', 'is_labour_day', 'is_thanksgiving', 'is_christmas',
                'is_pre_thanksgiving', 'is_pre_christmas', 'weeks_to_next_holiday'],
    'lag': ['sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_12', 'sales_lag_52'],
    'rolling': ['rolling_mean_4w', 'rolling_mean_8w', 'rolling_mean_12w',
                'rolling_std_4w', 'rolling_std_8w', 'ewma_4w', 'ewma_8w'],
    'store': ['store_mean_sales', 'store_std_sales', 'store_rank'],
    'economic': ['temp_deviation', 'fuel_change', 'cpi_change', 'unemployment_change']
}

# Model hyperparameter search spaces for Optuna
OPTUNA_TRIALS = 20  # Quick mode

RF_PARAM_SPACE = {
    'n_estimators': (100, 300),
    'max_depth': (5, 20),
    'min_samples_split': (2, 15),
    'min_samples_leaf': (1, 8),
}

XGB_PARAM_SPACE = {
    'n_estimators': (100, 300),
    'max_depth': (3, 12),
    'learning_rate': (0.01, 0.2),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'min_child_weight': (1, 10),
}

LGB_PARAM_SPACE = {
    'n_estimators': (100, 300),
    'max_depth': (3, 15),
    'learning_rate': (0.01, 0.2),
    'num_leaves': (20, 100),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
}

# Target column
TARGET = 'Weekly_Sales'

# Original features from dataset
ORIGINAL_FEATURES = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
