"""
Evaluation metrics for Walmart Sales Forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error.

    WMAPE = SUM(|actual - predicted|) / SUM(actual) * 100

    This weights errors by actual values, giving more importance to
    high-sales periods which contribute more to business value.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        WMAPE as percentage (0-100)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Arrays must have same length. Got {len(y_true)} and {len(y_pred)}")

    total_actual = np.sum(y_true)
    if total_actual == 0:
        raise ValueError("Sum of actual values cannot be zero")

    absolute_errors = np.abs(y_true - y_pred)
    wmape_value = (np.sum(absolute_errors) / total_actual) * 100

    return wmape_value


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    MAPE = mean(|actual - predicted| / actual) * 100

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        MAPE as percentage (0-100)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        raise ValueError("All actual values are zero")

    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    mape_value = np.mean(percentage_errors) * 100

    return mape_value


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    SMAPE = mean(2 * |actual - predicted| / (|actual| + |predicted|)) * 100

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        SMAPE as percentage (0-100)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0

    smape_values = 2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    smape_value = np.mean(smape_values) * 100

    return smape_value


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Dictionary with all metric values
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    metrics = {
        'WMAPE': wmape(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'SMAPE': smape(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Mean_Error': np.mean(y_pred - y_true),
        'Median_AE': np.median(np.abs(y_true - y_pred))
    }

    return metrics


def wmape_per_store(
    df: pd.DataFrame,
    y_true_col: str = 'Weekly_Sales',
    y_pred_col: str = 'Predicted'
) -> pd.DataFrame:
    """
    Calculate WMAPE for each store separately.

    Args:
        df: DataFrame with Store, actual, and predicted columns
        y_true_col: Name of actual values column
        y_pred_col: Name of predicted values column

    Returns:
        DataFrame with Store and WMAPE columns
    """
    results = []
    for store in sorted(df['Store'].unique()):
        store_df = df[df['Store'] == store]
        try:
            store_wmape = wmape(store_df[y_true_col], store_df[y_pred_col])
            results.append({'Store': store, 'WMAPE': store_wmape})
        except Exception as e:
            results.append({'Store': store, 'WMAPE': np.nan})

    return pd.DataFrame(results)


def wmape_per_period(
    df: pd.DataFrame,
    y_true_col: str = 'Weekly_Sales',
    y_pred_col: str = 'Predicted',
    period_col: str = 'Holiday_Flag'
) -> pd.DataFrame:
    """
    Calculate WMAPE for different periods (e.g., holiday vs non-holiday).

    Args:
        df: DataFrame with period indicator and predictions
        y_true_col: Name of actual values column
        y_pred_col: Name of predicted values column
        period_col: Name of period indicator column

    Returns:
        DataFrame with period and WMAPE
    """
    results = []
    for period in df[period_col].unique():
        period_df = df[df[period_col] == period]
        try:
            period_wmape = wmape(period_df[y_true_col], period_df[y_pred_col])
            results.append({
                period_col: period,
                'WMAPE': period_wmape,
                'Count': len(period_df)
            })
        except Exception as e:
            results.append({
                period_col: period,
                'WMAPE': np.nan,
                'Count': len(period_df)
            })

    return pd.DataFrame(results)


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Dictionary of metric names and values

    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if name in ['WMAPE', 'MAPE', 'SMAPE']:
            lines.append(f"  {name}: {value:.2f}%")
        elif name in ['MAE', 'RMSE', 'Mean_Error', 'Median_AE']:
            lines.append(f"  {name}: ${value:,.0f}")
        elif name == 'R2':
            lines.append(f"  {name}: {value:.4f}")
        else:
            lines.append(f"  {name}: {value:.4f}")

    return "\n".join(lines)


def compare_models(
    results: Dict[str, Dict[str, float]],
    primary_metric: str = 'WMAPE'
) -> pd.DataFrame:
    """
    Compare multiple models and rank by primary metric.

    Args:
        results: Dictionary of {model_name: metrics_dict}
        primary_metric: Metric to sort by (lower is better)

    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(results).T
    df = df.sort_values(primary_metric)
    df['Rank'] = range(1, len(df) + 1)

    # Reorder columns
    cols = ['Rank', primary_metric] + [c for c in df.columns if c not in ['Rank', primary_metric]]
    df = df[cols]

    return df


class MetricsTracker:
    """
    Track metrics during training for visualization.
    """

    def __init__(self):
        self.history = []

    def log(self, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """Log metrics for an epoch."""
        entry = {'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()}}
        if val_metrics:
            entry.update({f'val_{k}': v for k, v in val_metrics.items()})
        self.history.append(entry)

    def get_history(self) -> pd.DataFrame:
        """Get metrics history as DataFrame."""
        return pd.DataFrame(self.history)

    def best_epoch(self, metric: str = 'val_WMAPE') -> int:
        """Get epoch with best metric value."""
        df = self.get_history()
        if metric not in df.columns:
            return -1
        return df[metric].idxmin()


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    y_true = np.random.uniform(500000, 2000000, 100)
    y_pred = y_true * np.random.uniform(0.9, 1.1, 100)

    print("Metrics Test:")
    metrics = calculate_all_metrics(y_true, y_pred)
    print(format_metrics(metrics))
