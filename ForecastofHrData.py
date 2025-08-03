import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Title and disclaimer
st.title("Power Demand Forecasting and Analysis")
st.markdown("**Copyright © 2025, NITI Aayog**")
st.markdown("This tool helps forecast power demand using various statistical models and provides financial implications based on predictions.")

# Sidebar inputs
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Statistical Model", 
                                    ["Linear Regression", "Random Forest", "SVR", "XGBoost", "SARIMAX"])
train_percent = st.sidebar.slider("Training Data Percentage", 0, 100, 70)

# File uploader
uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])

if uploaded_file:
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name="Yearly Demand Profile", engine="openpyxl")
    df['DateTime'] = pd.to_datetime(df['DateTime'] + ' ' + df['Year'].astype(str), format='%d-%b %I%p %Y')
    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Prepare data
    data = df['Power Demand (MW)'].values
    n = len(data)
    train_size = int(n * train_percent / 100)
    train, test = data[:train_size], data[train_size:]
    X_train = np.arange(train_size).reshape(-1, 1)
    X_test = np.arange(train_size, n).reshape(-1, 1)

    # Model training and prediction
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, train)
        predictions = model.predict(X_test)
    elif model_choice == "Random Forest":
        model = RandomForestRegressor()
        model.fit(X_train, train)
        predictions = model.predict(X_test)
    elif model_choice == "SVR":
        model = SVR()
        model.fit(X_train, train)
        predictions = model.predict(X_test)
    elif model_choice == "XGBoost":
        model = xgb.XGBRegressor()
        model.fit(X_train, train)
        predictions = model.predict(X_test)
    elif model_choice == "SARIMAX":
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,24))
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=len(test))

    # Metrics
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))

    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"R² Score: {r2:.4f}")
    st.sidebar.write(f"MAE: {mae:.2f}")
    st.sidebar.write(f"RMSE: {rmse:.2f}")

    # Insights
    st.sidebar.subheader("Model Insights")
    insights = []
    if r2 > 0.8:
        insights.append("High accuracy in predictions.")
    elif r2 > 0.5:
        insights.append("Moderate accuracy. Consider tuning parameters.")
    else:
        insights.append("Low accuracy. Model may not be suitable.")
    if mae < 10000:
        insights.append("Low average error in predictions.")
    else:
        insights.append("High average error. Consider alternative models.")
    for insight in insights:
        st.sidebar.markdown(f"- {insight}")

    # Financial implications
    savings = np.maximum(0, test - predictions)
    total_savings_mw = np.sum(savings)
    daily_savings_inr = np.mean(savings) * 4000
    yearly_savings_inr = total_savings_mw * 4000

    st.subheader("Financial Implications")
    st.write(f"Average Daily Savings: ₹{daily_savings_inr:,.2f}")
    st.write(f"Estimated Yearly Savings: ₹{yearly_savings_inr:,.2f}")
    st.markdown("_Disclaimer: The average cost per MW considered is INR 4000._")

    # Visualization
    st.subheader("Prediction vs Actuals vs Baseline")
    baseline = np.mean(train)
    plt.figure(figsize=(12,6))
    plt.plot(df['DateTime'][train_size:], test, label='Actual')
    plt.plot(df['DateTime'][train_size:], predictions, label='Predicted')
    plt.axhline(y=baseline, color='gray', linestyle='--', label='Baseline Mean')
    plt.xlabel("DateTime")
    plt.ylabel("Power Demand (MW)")
    plt.legend()
    st.pyplot(plt)
