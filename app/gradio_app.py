"""
Gradio WebUI for Walmart Sales Forecasting.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
import joblib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PATH, MODELS_DIR, TARGET, HOLIDAYS
from src.data_preprocessing import load_data, time_based_split, get_data_summary
from src.feature_engineering import engineer_features, engineer_features_split, handle_missing_features
from src.metrics import wmape, calculate_all_metrics, wmape_per_store
from src.config import VALIDATION_START_DATE


# Global variables for data and models
DATA = None
TRAIN_DF = None
VAL_DF = None
MODELS = {}
PREDICTIONS_DF = None
METRICS = {}
FEATURE_COLS = None
SIMPLE_MODEL = None
SIMPLE_FEATURE_COLS = None


def load_all_data():
    """Load data and trained models."""
    global DATA, TRAIN_DF, VAL_DF, MODELS, PREDICTIONS_DF, METRICS, FEATURE_COLS
    global SIMPLE_MODEL, SIMPLE_FEATURE_COLS

    print("Loading data...")
    DATA = load_data()
    TRAIN_DF, VAL_DF = time_based_split(DATA)

    # Load models if available
    print("Loading models...")
    model_files = list(MODELS_DIR.glob("*_model.joblib"))
    for model_file in model_files:
        model_name = model_file.stem.replace("_model", "").title()
        if model_name == "Lightgbm":
            model_name = "LightGBM"
        if model_name == "Simple":
            continue  # Handle separately
        try:
            MODELS[model_name] = joblib.load(model_file)
            print(f"Loaded {model_name}")
        except Exception as e:
            print(f"Could not load {model_name}: {e}")

    # Load simple model for predictions
    simple_model_path = MODELS_DIR / "simple_model.joblib"
    simple_features_path = MODELS_DIR / "simple_feature_cols.joblib"
    if simple_model_path.exists():
        SIMPLE_MODEL = joblib.load(simple_model_path)
        print("Loaded Simple model for predictions")
    if simple_features_path.exists():
        SIMPLE_FEATURE_COLS = joblib.load(simple_features_path)

    # Load feature columns
    feature_cols_path = MODELS_DIR / "feature_columns.joblib"
    if feature_cols_path.exists():
        FEATURE_COLS = joblib.load(feature_cols_path)

    # Load predictions if available
    predictions_path = MODELS_DIR / "results" / "predictions.csv"
    if predictions_path.exists():
        PREDICTIONS_DF = pd.read_csv(predictions_path)
        PREDICTIONS_DF['Date'] = pd.to_datetime(PREDICTIONS_DF['Date'])

    # Load metrics if available
    metrics_path = MODELS_DIR / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            METRICS = json.load(f)

    print("Data loading complete!")


# ============= EDA CHARTS =============

def plot_sales_distribution():
    """Create sales distribution histogram."""
    fig = px.histogram(
        DATA, x='Weekly_Sales', nbins=50,
        title='Distribution of Weekly Sales',
        labels={'Weekly_Sales': 'Weekly Sales ($)'},
        color_discrete_sequence=['#3498db']
    )

    mean_sales = DATA['Weekly_Sales'].mean()
    median_sales = DATA['Weekly_Sales'].median()

    fig.add_vline(x=mean_sales, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: ${mean_sales:,.0f}")
    fig.add_vline(x=median_sales, line_dash="dot", line_color="green",
                  annotation_text=f"Median: ${median_sales:,.0f}")

    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_sales_over_time(store_id=None):
    """Create sales over time line chart."""
    if store_id and store_id != "All Stores":
        store_id = int(store_id)
        df = DATA[DATA['Store'] == store_id].copy()
        title = f'Weekly Sales Over Time - Store {store_id}'
    else:
        df = DATA.groupby('Date')['Weekly_Sales'].mean().reset_index()
        title = 'Average Weekly Sales Over Time (All Stores)'

    fig = px.line(
        df, x='Date', y='Weekly_Sales',
        title=title,
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Date': 'Date'}
    )

    # Add holiday markers
    for holiday_name, dates in HOLIDAYS.items():
        for date in dates:
            if date >= DATA['Date'].min() and date <= DATA['Date'].max():
                fig.add_vline(x=date, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_sales_by_store():
    """Create bar chart of average sales by store."""
    store_avg = DATA.groupby('Store')['Weekly_Sales'].mean().reset_index()
    store_avg = store_avg.sort_values('Weekly_Sales', ascending=False)

    fig = px.bar(
        store_avg, x='Store', y='Weekly_Sales',
        title='Average Weekly Sales by Store',
        labels={'Weekly_Sales': 'Avg Weekly Sales ($)', 'Store': 'Store ID'},
        color='Weekly_Sales',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_holiday_impact():
    """Create box plot comparing holiday vs non-holiday sales."""
    df = DATA.copy()
    df['Holiday'] = df['Holiday_Flag'].map({0: 'Non-Holiday', 1: 'Holiday'})

    fig = px.box(
        df, x='Holiday', y='Weekly_Sales',
        title='Sales Distribution: Holiday vs Non-Holiday Weeks',
        labels={'Weekly_Sales': 'Weekly Sales ($)', 'Holiday': ''},
        color='Holiday',
        color_discrete_map={'Non-Holiday': '#3498db', 'Holiday': '#e74c3c'}
    )

    fig.update_layout(template='plotly_white', height=400, showlegend=False)
    return fig


def plot_seasonality():
    """Create monthly seasonality pattern chart."""
    df = DATA.copy()
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.month_name()

    monthly_avg = df.groupby(['Month', 'Month_Name'])['Weekly_Sales'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values('Month')

    fig = px.bar(
        monthly_avg, x='Month_Name', y='Weekly_Sales',
        title='Average Sales by Month (Seasonality)',
        labels={'Weekly_Sales': 'Avg Weekly Sales ($)', 'Month_Name': 'Month'},
        color='Weekly_Sales',
        color_continuous_scale='RdYlGn'
    )

    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_correlation_matrix():
    """Create feature correlation heatmap."""
    numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI',
                    'Unemployment', 'Holiday_Flag']
    corr = DATA[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto='.2f',
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )

    fig.update_layout(template='plotly_white', height=450)
    return fig


def plot_store_heatmap():
    """Create store x month sales heatmap."""
    df = DATA.copy()
    df['Month'] = df['Date'].dt.month

    pivot = df.pivot_table(
        values='Weekly_Sales',
        index='Store',
        columns='Month',
        aggfunc='mean'
    )

    fig = px.imshow(
        pivot,
        title='Average Sales by Store and Month',
        labels={'x': 'Month', 'y': 'Store', 'color': 'Avg Sales'},
        color_continuous_scale='Viridis',
        aspect='auto'
    )

    fig.update_layout(template='plotly_white', height=600)
    return fig


# ============= RESULTS CHARTS =============

def get_metrics_table():
    """Create metrics comparison table."""
    if not METRICS:
        return pd.DataFrame({'Message': ['No trained models found. Run train.py first.']})

    df = pd.DataFrame(METRICS).T
    df = df.round(2)
    df = df.sort_values('WMAPE')
    df = df.reset_index()
    df.columns = ['Model'] + list(df.columns[1:])

    # Format columns
    format_cols = ['WMAPE', 'MAPE', 'SMAPE']
    for col in format_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%")

    money_cols = ['MAE', 'RMSE', 'Mean_Error', 'Median_AE']
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"${x:,.0f}")

    if 'R2' in df.columns:
        df['R2'] = df['R2'].apply(lambda x: f"{float(x.replace('%', '')):.4f}" if isinstance(x, str) else f"{x:.4f}")

    return df[['Model', 'WMAPE', 'MAE', 'RMSE', 'R2']]


def plot_actual_vs_predicted(model_name=None):
    """Create actual vs predicted comparison chart."""
    if PREDICTIONS_DF is None:
        fig = go.Figure()
        fig.add_annotation(text="No predictions found. Run train.py first.",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = PREDICTIONS_DF.copy()

    # Aggregate by date for cleaner visualization
    agg_df = df.groupby('Date').agg({
        TARGET: 'sum',
        **{col: 'sum' for col in df.columns if '_Pred' in col}
    }).reset_index()

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=agg_df['Date'], y=agg_df[TARGET],
        mode='lines', name='Actual',
        line=dict(color='#2c3e50', width=2)
    ))

    # Predictions
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']
    pred_cols = [col for col in agg_df.columns if '_Pred' in col]

    for i, col in enumerate(pred_cols):
        model = col.replace('_Pred', '')
        if model_name is None or model_name == "All Models" or model == model_name:
            fig.add_trace(go.Scatter(
                x=agg_df['Date'], y=agg_df[col],
                mode='lines', name=model,
                line=dict(color=colors[i % len(colors)], width=1.5, dash='dash')
            ))

    fig.update_layout(
        title='Actual vs Predicted Sales (Aggregated)',
        xaxis_title='Date',
        yaxis_title='Total Weekly Sales ($)',
        template='plotly_white',
        height=450,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def plot_wmape_by_store(model_name=None):
    """Create WMAPE by store bar chart."""
    if PREDICTIONS_DF is None:
        fig = go.Figure()
        fig.add_annotation(text="No predictions found. Run train.py first.",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = PREDICTIONS_DF.copy()

    # Calculate WMAPE per store for each model
    pred_cols = [col for col in df.columns if '_Pred' in col]

    if model_name and model_name != "All Models":
        pred_cols = [col for col in pred_cols if model_name in col]

    store_wmapes = []
    for store in sorted(df['Store'].unique()):
        store_df = df[df['Store'] == store]
        row = {'Store': store}
        for col in pred_cols:
            model = col.replace('_Pred', '')
            try:
                row[model] = wmape(store_df[TARGET], store_df[col])
            except:
                row[model] = np.nan
        store_wmapes.append(row)

    wmape_df = pd.DataFrame(store_wmapes)

    # Use best model if not specified
    if model_name is None or model_name == "All Models":
        # Find model with lowest average WMAPE
        avg_wmapes = {col: wmape_df[col].mean() for col in wmape_df.columns if col != 'Store'}
        best_model = min(avg_wmapes, key=avg_wmapes.get)
    else:
        best_model = model_name

    if best_model in wmape_df.columns:
        fig = px.bar(
            wmape_df, x='Store', y=best_model,
            title=f'WMAPE by Store ({best_model})',
            labels={best_model: 'WMAPE (%)', 'Store': 'Store ID'},
            color=best_model,
            color_continuous_scale='RdYlGn_r'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(text="Model not found", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)

    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_residuals(model_name=None):
    """Create residual distribution histogram."""
    if PREDICTIONS_DF is None:
        fig = go.Figure()
        fig.add_annotation(text="No predictions found. Run train.py first.",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = PREDICTIONS_DF.copy()

    pred_cols = [col for col in df.columns if '_Pred' in col]
    if model_name and model_name != "All Models":
        pred_cols = [col for col in pred_cols if model_name in col]

    if not pred_cols:
        fig = go.Figure()
        fig.add_annotation(text="Model not found", xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

    # Use first matching model
    pred_col = pred_cols[0]
    model = pred_col.replace('_Pred', '')

    residuals = df[TARGET] - df[pred_col]

    fig = px.histogram(
        x=residuals, nbins=50,
        title=f'Residual Distribution ({model})',
        labels={'x': 'Residual (Actual - Predicted)', 'y': 'Count'},
        color_discrete_sequence=['#3498db']
    )

    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.add_vline(x=residuals.mean(), line_dash="dot", line_color="green",
                  annotation_text=f"Mean: ${residuals.mean():,.0f}")

    fig.update_layout(template='plotly_white', height=400)
    return fig


def plot_feature_importance(model_name=None):
    """Create feature importance bar chart."""
    if not MODELS or not FEATURE_COLS:
        fig = go.Figure()
        fig.add_annotation(text="No trained models found. Run train.py first.",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # Find a model with feature importance
    model = None
    for name in ['Xgboost', 'XGBoost', 'Lightgbm', 'LightGBM', 'Randomforest', 'RandomForest']:
        if name in MODELS:
            model = MODELS[name]
            model_name = name
            break

    if model is None or not hasattr(model, 'feature_importance_'):
        fig = go.Figure()
        fig.add_annotation(text="No feature importance available",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    importance = model.feature_importance_
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLS[:len(importance)],
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=True).tail(20)

    fig = px.bar(
        importance_df, x='Importance', y='Feature',
        orientation='h',
        title=f'Top 20 Feature Importance ({model_name})',
        labels={'Importance': 'Importance Score', 'Feature': ''},
        color='Importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(template='plotly_white', height=500)
    return fig


# ============= PREDICTION FUNCTION =============

def make_prediction(store, date_str, temperature, fuel_price, cpi, unemployment, is_holiday):
    """Make a sales prediction using the simple model (no lag features)."""
    if SIMPLE_MODEL is None:
        return "Simple prediction model not available. Please run training first."

    try:
        # Parse date
        date = pd.to_datetime(date_str, format='%d-%m-%Y')
    except:
        return "Invalid date format. Please use DD-MM-YYYY format."

    try:
        store_id = int(store)
        temp_val = float(temperature)
        fuel_val = float(fuel_price)
        cpi_val = float(cpi)
        unemp_val = float(unemployment)
        holiday_val = 1 if is_holiday else 0

        # Build features for simple model (no lags!)
        features = {
            'Store': store_id,
            'Holiday_Flag': holiday_val,
            'Temperature': temp_val,
            'Fuel_Price': fuel_val,
            'CPI': cpi_val,
            'Unemployment': unemp_val,
            'year': date.year,
            'month': date.month,
            'week_of_year': date.isocalendar()[1],
            'quarter': (date.month - 1) // 3 + 1,
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'week_sin': np.sin(2 * np.pi * date.isocalendar()[1] / 52),
            'week_cos': np.cos(2 * np.pi * date.isocalendar()[1] / 52),
            'temp_squared': temp_val ** 2,
            'cpi_unemployment': cpi_val * unemp_val
        }

        # Build feature vector in correct order
        X = np.array([[features[col] for col in SIMPLE_FEATURE_COLS]])

        # Predict
        prediction = SIMPLE_MODEL.predict(X)[0]

        # Get historical context
        store_data = DATA[DATA['Store'] == store_id]
        avg_sales = store_data['Weekly_Sales'].mean()
        max_sales = store_data['Weekly_Sales'].max()
        min_sales = store_data['Weekly_Sales'].min()

        # Calculate difference from average
        diff_from_avg = prediction - avg_sales
        pct_diff = (diff_from_avg / avg_sales) * 100

        # Determine if prediction is high/low/normal
        if pct_diff > 10:
            trend_emoji = "📈"
            trend_text = "Above Average"
        elif pct_diff < -10:
            trend_emoji = "📉"
            trend_text = "Below Average"
        else:
            trend_emoji = "➡️"
            trend_text = "Near Average"

        result = f"""
## 🎯 Predicted Weekly Sales

# ${prediction:,.0f}

{trend_emoji} **{trend_text}** ({pct_diff:+.1f}% vs historical average)

---

### Prediction Breakdown:
| Metric | Value |
|--------|-------|
| **Your Prediction** | **${prediction:,.0f}** |
| Store {store_id} Average | ${avg_sales:,.0f} |
| Difference | ${diff_from_avg:+,.0f} ({pct_diff:+.1f}%) |

---

### Your Input Parameters:
| Parameter | Value |
|-----------|-------|
| Store | {store_id} |
| Date | {date.strftime('%d-%m-%Y')} |
| Temperature | {temp_val}°F |
| Fuel Price | ${fuel_val:.2f} |
| CPI | {cpi_val} |
| Unemployment | {unemp_val}% |
| Holiday Week | {'Yes ✓' if holiday_val else 'No'} |

---

<small>Store {store_id} historical range: ${min_sales:,.0f} - ${max_sales:,.0f}</small>
"""
        return result

    except Exception as e:
        import traceback
        return f"Prediction error: {str(e)}\n\n{traceback.format_exc()}"


# ============= GRADIO APP =============

def create_app():
    """Create the Gradio app."""

    # Load data on startup
    load_all_data()

    with gr.Blocks(title="Walmart Sales Forecasting", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # Walmart Sales Forecasting Dashboard

        This dashboard provides exploratory data analysis, model performance results,
        and sales predictions for Walmart stores.
        """)

        with gr.Tabs():
            # ===== TAB 1: EDA =====
            with gr.Tab("Exploratory Data Analysis"):
                gr.Markdown("## Data Overview")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""
                        **Dataset Statistics:**
                        - Total Records: {len(DATA):,}
                        - Number of Stores: {DATA['Store'].nunique()}
                        - Date Range: {DATA['Date'].min().strftime('%Y-%m-%d')} to {DATA['Date'].max().strftime('%Y-%m-%d')}
                        - Holiday Weeks: {DATA['Holiday_Flag'].sum()} ({DATA['Holiday_Flag'].mean()*100:.1f}%)
                        """)
                    with gr.Column():
                        gr.Markdown(f"""
                        **Sales Statistics:**
                        - Mean: ${DATA['Weekly_Sales'].mean():,.0f}
                        - Median: ${DATA['Weekly_Sales'].median():,.0f}
                        - Min: ${DATA['Weekly_Sales'].min():,.0f}
                        - Max: ${DATA['Weekly_Sales'].max():,.0f}
                        """)

                with gr.Row():
                    with gr.Column():
                        gr.Plot(plot_sales_distribution, label="Sales Distribution")
                    with gr.Column():
                        store_dropdown = gr.Dropdown(
                            choices=["All Stores"] + [str(i) for i in range(1, 46)],
                            value="All Stores",
                            label="Select Store"
                        )
                        sales_time_plot = gr.Plot(label="Sales Over Time")

                        store_dropdown.change(
                            plot_sales_over_time,
                            inputs=[store_dropdown],
                            outputs=[sales_time_plot]
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Plot(plot_sales_by_store, label="Sales by Store")
                    with gr.Column():
                        gr.Plot(plot_holiday_impact, label="Holiday Impact")

                with gr.Row():
                    with gr.Column():
                        gr.Plot(plot_seasonality, label="Monthly Seasonality")
                    with gr.Column():
                        gr.Plot(plot_correlation_matrix, label="Feature Correlations")

                gr.Markdown("## Store Performance Heatmap")
                gr.Plot(plot_store_heatmap, label="Store x Month Heatmap")

            # ===== TAB 2: RESULTS =====
            with gr.Tab("Model Results"):
                gr.Markdown("## Model Performance Comparison")

                metrics_table = gr.Dataframe(
                    value=get_metrics_table,
                    label="Model Metrics (sorted by WMAPE)"
                )

                with gr.Row():
                    model_selector = gr.Dropdown(
                        choices=["All Models"] + list(METRICS.keys()) if METRICS else ["All Models"],
                        value="All Models",
                        label="Select Model"
                    )

                with gr.Row():
                    with gr.Column():
                        actual_pred_plot = gr.Plot(label="Actual vs Predicted")
                    with gr.Column():
                        wmape_store_plot = gr.Plot(label="WMAPE by Store")

                with gr.Row():
                    with gr.Column():
                        residual_plot = gr.Plot(label="Residual Distribution")
                    with gr.Column():
                        importance_plot = gr.Plot(label="Feature Importance")

                # Update plots on model selection
                model_selector.change(
                    plot_actual_vs_predicted,
                    inputs=[model_selector],
                    outputs=[actual_pred_plot]
                )
                model_selector.change(
                    plot_wmape_by_store,
                    inputs=[model_selector],
                    outputs=[wmape_store_plot]
                )
                model_selector.change(
                    plot_residuals,
                    inputs=[model_selector],
                    outputs=[residual_plot]
                )

                # Initial plots
                app.load(plot_actual_vs_predicted, outputs=[actual_pred_plot])
                app.load(plot_wmape_by_store, outputs=[wmape_store_plot])
                app.load(plot_residuals, outputs=[residual_plot])
                app.load(plot_feature_importance, outputs=[importance_plot])

            # ===== TAB 3: PREDICTIONS =====
            with gr.Tab("Make Predictions"):
                gr.Markdown("""
                <div style="background-color: #ff9800; color: black; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <strong>⚠️ WORK IN PROGRESS</strong> - This prediction tool uses a simplified model (without lag features)
                to demonstrate how input parameters affect sales predictions. For production forecasting,
                use the full models shown in the Results tab which achieve ~1.14% WMAPE.
                </div>
                """)
                gr.Markdown("## Sales Prediction")
                gr.Markdown("Enter the details below to get a sales prediction. Changes to inputs will be reflected in the output.")

                with gr.Row():
                    with gr.Column():
                        pred_store = gr.Dropdown(
                            choices=[str(i) for i in range(1, 46)],
                            value="1",
                            label="Store"
                        )
                        pred_date = gr.Textbox(
                            label="Date (DD-MM-YYYY)",
                            placeholder="e.g., 15-10-2012",
                            value="15-10-2012"
                        )
                        pred_holiday = gr.Checkbox(label="Is Holiday Week?", value=False)

                    with gr.Column():
                        pred_temp = gr.Slider(
                            minimum=-10, maximum=110,
                            value=60, step=1,
                            label="Temperature (F)"
                        )
                        pred_fuel = gr.Number(label="Fuel Price ($)", value=3.5)

                    with gr.Column():
                        pred_cpi = gr.Number(label="CPI", value=210)
                        pred_unemployment = gr.Number(label="Unemployment (%)", value=8.0)

                predict_btn = gr.Button("Predict Sales", variant="primary")
                prediction_output = gr.Markdown("### Enter values and click Predict")

                predict_btn.click(
                    make_prediction,
                    inputs=[pred_store, pred_date, pred_temp, pred_fuel,
                           pred_cpi, pred_unemployment, pred_holiday],
                    outputs=[prediction_output]
                )

            # ===== TAB 4: ABOUT =====
            with gr.Tab("About"):
                gr.Markdown("""
                ## About This Dashboard

                This dashboard is part of the **Walmart Sales Forecasting** project,
                which predicts weekly sales for 45 Walmart stores using various machine learning
                and time series models.

                ### Models Used:
                - **Random Forest**: Ensemble of decision trees
                - **XGBoost**: Gradient boosted trees
                - **LightGBM**: Fast gradient boosting
                - **SARIMA**: Seasonal ARIMA time series model
                - **Prophet**: Facebook's time series forecasting tool
                - **Ensemble**: Weighted combination of all models

                ### Features Engineered:
                - Temporal features (month, week, quarter, etc.)
                - Cyclical features (sine/cosine encoding)
                - Holiday indicators and proximity features
                - Lag features (1, 2, 4, 12, 52 weeks)
                - Rolling statistics (4, 8, 12, 26 weeks)
                - Store-level aggregates
                - Economic indicator transformations

                ### Evaluation Metric:
                **WMAPE (Weighted Mean Absolute Percentage Error)**
                - Weights errors by actual sales volume
                - Lower is better

                ### How to Use:
                1. **EDA Tab**: Explore the data through various visualizations
                2. **Results Tab**: Compare model performance
                3. **Predictions Tab**: Make new sales predictions

                ---
                Built with Python, Gradio, and Plotly
                """)

        # Initialize time series plot
        app.load(lambda: plot_sales_over_time("All Stores"), outputs=[sales_time_plot])

    return app


def main():
    """Launch the Gradio app."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
