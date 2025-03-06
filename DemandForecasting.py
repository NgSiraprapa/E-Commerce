import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import math

# Load and preprocess dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    sales_data = df.groupby('Order Date')['Purchase Amount (USD)'].sum().reset_index()
    return sales_data

# Streamlit App
st.title("Demand Forecasting Dashboard")

# Load Data
file_path = "E-Commerce_Analytics_Dataset_Term Project.csv"
sales_data = load_data(file_path)

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

# Train XGBoost Regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Generate future dates (next 6 months)
last_date = sales_data['Order Date'].max()
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 181)]  # 6 months = 180 days
future_df = pd.DataFrame({'Date': [date.toordinal() for date in future_dates],
                          'month': [date.month for date in future_dates],
                          'day': [date.day for date in future_dates],
                          'weekday': [date.weekday() for date in future_dates]})

# Add seasonality features
future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)

# Predict future sales
predictions = model.predict(future_df[X.columns])

# Create forecast dataframe
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predictions})

# Tabs for visualization
tab1, tab2 = st.tabs(["Historical Data", "Forecast"])

with tab1:
    st.subheader("Sales Over Time")
    st.plotly_chart(px.line(sales_data, x='Order Date', y='Purchase Amount (USD)', title='Historical Sales Trend'))

with tab2:
    st.subheader("Predicted Sales (Next 6 Months)")
    st.plotly_chart(px.line(forecast_df, x='Date', y='Predicted Sales', title='Sales Forecast with Seasonality Adjustments'))

st.write("The dashboard predicts future demand for the next 6 months based on historical e-commerce sales data using XGBoost Regressor with seasonality adjustments.")