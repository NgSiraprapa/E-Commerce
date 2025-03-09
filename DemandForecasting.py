import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math
import os

# Load and preprocess dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}. Please check the file path and ensure the dataset exists.")
        return None
    df = pd.read_csv(file_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    sales_data = df.groupby('Order Date')['Purchase Amount (USD)'].sum().reset_index()
    return sales_data

# Streamlit App
st.title("📈 Demand Forecasting Dashboard")
st.markdown("### 📊 Predict future sales trends using AI-powered models")

# Load Data
file_path = "E-Commerce_Analytics_Dataset_Term_Project.csv"
sales_data = load_data(file_path)

if sales_data is not None:
    # Feature Engineering: Convert dates to numerical features
    sales_data['Date'] = sales_data['Order Date'].map(pd.Timestamp.toordinal)
    sales_data['month'] = sales_data['Order Date'].dt.month
    sales_data['day'] = sales_data['Order Date'].dt.day
    sales_data['weekday'] = sales_data['Order Date'].dt.weekday
    sales_data['sin_month'] = np.sin(2 * np.pi * sales_data['month'] / 12)
    sales_data['cos_month'] = np.cos(2 * np.pi * sales_data['month'] / 12)
    X = sales_data[['Date', 'month', 'day', 'weekday', 'sin_month', 'cos_month']]
    y = sales_data['Purchase Amount (USD)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Models
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Generate future dates (next 6 months)
    last_date = sales_data['Order Date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 181)]
    future_df = pd.DataFrame({'Date': [date.toordinal() for date in future_dates],
                              'month': [date.month for date in future_dates],
                              'day': [date.day for date in future_dates],
                              'weekday': [date.weekday() for date in future_dates]})

    # Add seasonality features
    future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_features = future_df[X_train.columns]

    # Predict future sales using all models
    xgb_predictions = xgb_model.predict(future_features)
    rf_predictions = rf_model.predict(future_features)
    lr_predictions = lr_model.predict(future_features)

    # Evaluate model performance
    models = {'XGBoost': xgb_model, 'Random Forest': rf_model, 'Linear Regression': lr_model}
    eval_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        eval_results[name] = {'MAE': mae, 'RMSE': rmse}

    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'XGBoost Predictions': xgb_predictions,
        'Random Forest Predictions': rf_predictions,
        'Linear Regression Predictions': lr_predictions
    })

    # Display all information on a single page
    st.subheader("📌 Sales Over Time")
    st.plotly_chart(px.line(sales_data, x='Order Date', y='Purchase Amount (USD)', title='📉 Historical Sales Trend', markers=True))

    st.subheader("📌 Predicted Sales (Next 6 Months)")
    fig = px.line(forecast_df, x='Date', y=['XGBoost Predictions', 'Random Forest Predictions', 'Linear Regression Predictions'],
                  title='🔮 Sales Forecast Comparison', markers=True)
    st.plotly_chart(fig)

    st.subheader("📌 Model Performance Metrics")
    st.write("📊 **Performance of different forecasting models:**")
    st.dataframe(pd.DataFrame(eval_results).T.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}))

    st.success("✅ The dashboard predicts future demand for the next 6 months based on historical e-commerce sales data using AI-powered forecasting models!")
