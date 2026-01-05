# Walmart Sales Forecasting

A machine learning system for predicting weekly sales across 45 Walmart stores, featuring multiple models, comprehensive feature engineering, and an interactive Gradio dashboard.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Overview

This project predicts weekly sales for Walmart stores using historical data and external factors (temperature, fuel prices, CPI, unemployment). The system achieves **1.14% WMAPE** on validation data using an ensemble of gradient boosting models.

### Key Features
- 6 different models (RandomForest, XGBoost, LightGBM, SARIMA, Prophet, Ensemble)
- 83 engineered features including lag, rolling, and cyclical features
- Interactive Gradio web dashboard with EDA, model results, and predictions
- Time-based train/validation split to prevent data leakage

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WALMART SALES FORECASTING                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│              │    │                  │    │                  │
│  Walmart.csv │───▶│ Data Preprocessing│───▶│ Feature Engineering│
│  (Raw Data)  │    │                  │    │                  │
│              │    │ - Date parsing   │    │ - Temporal (8)   │
│  6,435 rows  │    │ - Sorting        │    │ - Cyclical (8)   │
│  8 columns   │    │ - Validation     │    │ - Holiday (10)   │
│              │    │ - Train/Val split│    │ - Lag (6)        │
└──────────────┘    └──────────────────┘    │ - Rolling (14)   │
                                            │ - Store (9)      │
                                            │ - Economic (8)   │
                                            │ - Interactions(3)│
                                            │                  │
                                            │ Total: 83 features│
                                            └────────┬─────────┘
                                                     │
                    ┌────────────────────────────────┴────────────────────────┐
                    │                                                         │
                    ▼                                                         ▼
        ┌───────────────────────┐                            ┌───────────────────────┐
        │   ML MODELS (Global)  │                            │ TIME SERIES (Per-Store)│
        │                       │                            │                       │
        │ ┌───────────────────┐ │                            │ ┌───────────────────┐ │
        │ │   RandomForest    │ │                            │ │     SARIMA        │ │
        │ │   WMAPE: 1.63%    │ │                            │ │   (45 models)     │ │
        │ └───────────────────┘ │                            │ └───────────────────┘ │
        │ ┌───────────────────┐ │                            │ ┌───────────────────┐ │
        │ │     XGBoost       │ │                            │ │     Prophet       │ │
        │ │   WMAPE: 1.21%    │ │                            │ │   (45 models)     │ │
        │ └───────────────────┘ │                            │ └───────────────────┘ │
        │ ┌───────────────────┐ │                            │                       │
        │ │    LightGBM       │ │                            └───────────────────────┘
        │ │   WMAPE: 1.20%    │ │
        │ └───────────────────┘ │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   ENSEMBLE MODEL      │
        │                       │
        │  XGBoost: 53%         │
        │  LightGBM: 47%        │
        │                       │
        │  WMAPE: 1.14%         │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    GRADIO WEBUI       │
        │                       │
        │ ┌───────────────────┐ │
        │ │  EDA Dashboard    │ │
        │ └───────────────────┘ │
        │ ┌───────────────────┐ │
        │ │  Model Results    │ │
        │ └───────────────────┘ │
        │ ┌───────────────────┐ │
        │ │  Predictions      │ │
        │ │  (Simple Model)   │ │
        │ └───────────────────┘ │
        └───────────────────────┘
```

### Data Flow

```
1. INPUT
   └── Walmart.csv (6,435 rows × 8 columns)

2. PREPROCESSING
   ├── Parse dates (DD-MM-YYYY → datetime)
   ├── Sort by Store, Date
   └── Time-based split:
       ├── Train: Feb 2010 - Aug 2012 (6,075 rows)
       └── Val: Sep - Oct 2012 (360 rows)

3. FEATURE ENGINEERING
   ├── Compute on full data (for lag features)
   ├── Split into train/val
   ├── Store features from train only (prevent leakage)
   └── Output: 83 numeric features

4. MODEL TRAINING
   ├── Optuna hyperparameter tuning (20 trials)
   ├── Train each model
   ├── Evaluate on validation set
   └── Optimize ensemble weights

5. OUTPUT
   ├── Saved models (models/*.joblib)
   ├── Predictions (models/results/predictions.csv)
   └── Metrics (models/results/metrics.json)
```

---

## Data

### Source
- **File**: `Walmart.csv`
- **Records**: 6,435 (45 stores × 143 weeks)
- **Date Range**: February 5, 2010 - October 26, 2012

### Columns

| Column | Type | Description |
|--------|------|-------------|
| Store | int | Store identifier (1-45) |
| Date | date | Week of sales (DD-MM-YYYY) |
| Weekly_Sales | float | Target variable - sales in dollars |
| Holiday_Flag | binary | 1 = Holiday week, 0 = Non-holiday |
| Temperature | float | Temperature in Fahrenheit |
| Fuel_Price | float | Regional fuel price |
| CPI | float | Consumer Price Index |
| Unemployment | float | Unemployment rate (%) |

### Holiday Events
- **Super Bowl**: Feb 12, 2010 | Feb 11, 2011 | Feb 10, 2012
- **Labor Day**: Sep 10, 2010 | Sep 9, 2011 | Sep 7, 2012
- **Thanksgiving**: Nov 26, 2010 | Nov 25, 2011 | Nov 23, 2012
- **Christmas**: Dec 31, 2010 | Dec 30, 2011 | Dec 28, 2012

### Data Quality
- No missing values
- No duplicate records
- Complete coverage (every store has data for all 143 weeks)

---

## Feature Engineering

### Feature Categories (83 total)

#### 1. Temporal Features (8)
```python
year, month, week_of_year, day_of_year, quarter,
week_of_month, is_month_start, is_month_end
```

#### 2. Cyclical Features (8)
Sine/cosine encoding to capture periodicity:
```python
month_sin, month_cos      # 12-month cycle
week_sin, week_cos        # 52-week cycle
day_of_year_sin/cos       # 365-day cycle
quarter_sin, quarter_cos  # 4-quarter cycle
```

#### 3. Holiday Features (10)
```python
is_super_bowl, is_labour_day, is_thanksgiving, is_christmas
is_pre_thanksgiving, is_pre_christmas, is_black_friday_week
weeks_to_next_holiday, weeks_since_last_holiday
Holiday_Flag (original)
```

#### 4. Lag Features (6)
Previous sales values per store:
```python
sales_lag_1   # 1 week ago
sales_lag_2   # 2 weeks ago
sales_lag_4   # 4 weeks ago
sales_lag_8   # 8 weeks ago
sales_lag_12  # 12 weeks ago
sales_lag_52  # Same week last year
```

#### 5. Rolling Features (14)
Window statistics per store:
```python
rolling_mean_4w, rolling_mean_8w, rolling_mean_12w, rolling_mean_26w
rolling_std_4w, rolling_std_8w
rolling_min_4w, rolling_max_4w
rolling_min_8w, rolling_max_8w
ewma_4w, ewma_8w, ewma_12w  # Exponential weighted moving average
```

#### 6. Store Features (9)
Aggregated from training data only:
```python
store_mean_sales, store_std_sales, store_median_sales
store_min_sales, store_max_sales
store_rank, store_percentile, store_cv
Store (original)
```

#### 7. Economic Features (8)
```python
Temperature, temp_deviation, temp_squared
Fuel_Price, fuel_change
CPI, cpi_change
Unemployment, unemployment_change, unemployment_above_avg
```

#### 8. Interaction Features (3)
```python
store_holiday    # Store × Holiday_Flag
store_month      # Store × month
cpi_unemployment # CPI × Unemployment
```

---

## Models

### Model Comparison

| Model | Type | WMAPE | MAE | RMSE | R² |
|-------|------|-------|-----|------|-----|
| **Ensemble** | Meta | **1.14%** | $11,470 | $16,978 | 0.9989 |
| LightGBM | Gradient Boosting | 1.20% | $12,060 | $17,409 | 0.9989 |
| XGBoost | Gradient Boosting | 1.21% | $12,205 | $18,320 | 0.9987 |
| RandomForest | Bagging | 1.63% | $16,362 | $25,687 | 0.9975 |
| Simple* | XGBoost (no lags) | 4.60% | - | - | 0.9769 |

*Simple model used for interactive predictions (responds to input changes)

### Ensemble Weights
Optimized on validation set to minimize WMAPE:
- **XGBoost**: 53%
- **LightGBM**: 47%
- RandomForest: 0% (excluded due to higher error)

### Hyperparameter Tuning
- **Method**: Optuna with 20 trials
- **Objective**: Minimize WMAPE
- **Validation**: Time-series cross-validation

### WMAPE Formula
```
WMAPE = Σ|actual - predicted| / Σ(actual) × 100
```
Weighted by actual sales, giving more importance to high-sales weeks.

---

## Results

### Validation Performance
- **Period**: September - October 2012 (last 2 months)
- **Samples**: 315 (after dropping rows with NaN lag features)
- **Best WMAPE**: 1.14% (Ensemble)

### Feature Importance (Top 10)
1. `store_mean_sales` - Store's historical average
2. `sales_lag_1` - Previous week's sales
3. `rolling_mean_4w` - 4-week rolling average
4. `Store` - Store identifier
5. `ewma_4w` - Exponential weighted average
6. `sales_lag_52` - Same week last year
7. `store_rank` - Store ranking by sales
8. `rolling_mean_8w` - 8-week rolling average
9. `week_of_year` - Seasonality
10. `month` - Monthly patterns

---

## Installation

### Prerequisites
- Python 3.10+
- uv (recommended) or pip

### Setup
```bash
# Clone repository
cd walmart_kaggle

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Or with pip
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
statsmodels>=0.14.0
optuna>=3.3.0
plotly>=5.17.0
gradio>=4.0.0
joblib>=1.3.0
tqdm>=4.66.0
```

---

## Usage

### Train Models
```bash
# Full training (all models, ~5-10 min)
python train.py

# Quick training (skip SARIMA/Prophet, ~30 sec)
python train.py --skip-ts

# Without hyperparameter tuning
python train.py --no-tune
```

### Launch Dashboard
```bash
python -m app.gradio_app
# Opens at http://localhost:7860
```

### Dashboard Features

#### Tab 1: Exploratory Data Analysis
- Sales distribution histogram
- Sales over time (by store)
- Average sales by store (bar chart)
- Holiday vs non-holiday comparison
- Monthly seasonality patterns
- Feature correlation heatmap
- Store × Month heatmap

#### Tab 2: Model Results
- Performance comparison table
- Actual vs Predicted chart
- WMAPE by store
- Residual distribution
- Feature importance (Top 20)

#### Tab 3: Make Predictions
- Interactive prediction form
- Uses simple model (no lag features) for responsiveness
- Shows prediction vs historical average
- Displays trend indicators

---

## Project Structure

```
walmart_kaggle/
├── Walmart.csv              # Source data
├── requirements.txt         # Python dependencies
├── train.py                 # Training pipeline
├── README.md                # This file
│
├── src/                     # Source modules
│   ├── __init__.py
│   ├── config.py           # Configuration & constants
│   │   ├── HOLIDAYS        # Holiday date definitions
│   │   ├── OPTUNA_TRIALS   # Tuning iterations (20)
│   │   └── *_PARAM_SPACE   # Model hyperparameter ranges
│   │
│   ├── data_preprocessing.py
│   │   ├── load_data()     # Load and parse CSV
│   │   ├── validate_data() # Check data integrity
│   │   └── time_based_split() # Train/val split
│   │
│   ├── feature_engineering.py
│   │   ├── create_temporal_features()
│   │   ├── create_cyclical_features()
│   │   ├── create_holiday_features()
│   │   ├── create_lag_features()
│   │   ├── create_rolling_features()
│   │   ├── create_store_features()
│   │   ├── create_economic_features()
│   │   ├── engineer_features()      # Main pipeline
│   │   └── engineer_features_split() # With proper lag handling
│   │
│   ├── metrics.py
│   │   ├── wmape()          # Primary metric
│   │   ├── mape(), smape()  # Alternative metrics
│   │   └── calculate_all_metrics()
│   │
│   └── models.py
│       ├── BaseModel        # Abstract base class
│       ├── RandomForestModel
│       ├── XGBoostModel
│       ├── LightGBMModel
│       ├── SARIMAModel      # Per-store fitting
│       ├── ProphetModel     # Per-store fitting
│       └── EnsembleModel    # Weight optimization
│
├── app/                     # Gradio application
│   ├── __init__.py
│   └── gradio_app.py       # Dashboard UI
│       ├── load_all_data()
│       ├── plot_* functions # EDA visualizations
│       ├── get_metrics_table()
│       └── make_prediction() # Interactive predictions
│
└── models/                  # Saved models & results
    ├── randomforest_model.joblib
    ├── xgboost_model.joblib
    ├── lightgbm_model.joblib
    ├── ensemble_model.joblib
    ├── simple_model.joblib  # For predictions tab
    ├── feature_columns.joblib
    ├── simple_feature_cols.joblib
    └── results/
        ├── metrics.json     # Model performance
        └── predictions.csv  # Validation predictions
```

---

## Technical Notes

### Preventing Data Leakage
1. **Time-based split**: Validation data is strictly after training data
2. **Lag features**: Computed on full data, then split (validation uses training history)
3. **Store features**: Statistics computed only from training data
4. **Target encoding**: Would use k-fold CV if implemented

### Why Two Models for Predictions?
- **Full model** (1.14% WMAPE): Uses lag features, best accuracy, but predictions don't respond to input changes since lags dominate
- **Simple model** (4.60% WMAPE): No lag features, responds to Temperature/CPI/etc. changes, better for interactive exploration

### Handling Missing Lag Values
- First ~52 weeks per store have NaN for `sales_lag_52`
- Strategy: Drop rows with NaN (results in 3,735 training samples from 6,075)
- Alternative: Fill with store mean (less accurate)

---

## License

MIT License

---

## Acknowledgments

- Data source: Walmart Sales Forecasting (Kaggle)
- Built with: scikit-learn, XGBoost, LightGBM, Gradio, Plotly
