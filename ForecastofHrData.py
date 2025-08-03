import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Title and disclaimer
st.title("Power Demand Forecasting and Analysis")
st.markdown("*(Data Copyright © 2025, NITI Aayog)*")
st.markdown("This tool helps forecast power demand using various statistical models and provides financial implications based on predictions.")

# Sidebar inputs
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Statistical Model", 
                                    ["Linear Regression", "Random Forest", "SVR", "XGBoost", "SARIMAX"])
train_percent = st.sidebar.slider("Training Data Percentage", 0, 100, 70)

# File uploader
uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])

def train_and_predict(data, train_percent, model_choice):
    n = len(data)
    train_size = int(n * train_percent / 100)
    train, test = data[:train_size], data[train_size:]
    X_train = np.arange(train_size).reshape(-1, 1)
    X_test = np.arange(train_size, n).reshape(-1, 1)

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

    return train, test, predictions

def calculate_metrics(test, predictions):
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    evs = explained_variance_score(test, predictions)
    return r2, mae, rmse, mape, mse, evs

def calculate_financials(test, predictions):
    savings = np.maximum(0, test - predictions)
    total_savings_mw = np.sum(savings)
    daily_savings_inr = np.mean(savings) * 4000
    yearly_savings_inr = total_savings_mw * 4000
    return daily_savings_inr, yearly_savings_inr

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, sheet_name="Yearly Demand Profile", engine="openpyxl")
        if 'DateTime' in df.columns and 'Year' in df.columns and 'Power Demand (MW)' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'] + ' ' + df['Year'].astype(str), format='%d-%b %I%p %Y')
            df.sort_values('DateTime', inplace=True)
            df.reset_index(drop=True, inplace=True)

            data = df['Power Demand (MW)'].values

            train, test, predictions = train_and_predict(data, train_percent, model_choice)
            r2, mae, rmse, mape, mse, evs = calculate_metrics(test, predictions)
            daily_savings_inr, yearly_savings_inr = calculate_financials(test, predictions)

            # Sidebar metrics
            st.sidebar.subheader("Statistical Highlights")
            st.sidebar.write(f"R² Score: {r2:.4f}")
            st.sidebar.write(f"MAE: {mae:.2f}")
            st.sidebar.write(f"RMSE: {rmse:.2f}")
            st.sidebar.write(f"MAPE: {mape:.2f}")
            st.sidebar.write(f"MSE: {mse:.2f}")
            st.sidebar.write(f"Explained Variance Score: {evs:.2f}")

            st.sidebar.subheader("Model Insights")
            insights = []
            if r2 > 0.8:
                insights.append("✅ High predictive accuracy.")
            elif r2 > 0.5:
                insights.append("⚠️ Moderate accuracy. May need tuning.")
            else:
                insights.append("❌ Low accuracy. Consider alternative models.")
            if mae < 10000:
                insights.append("✅ Low average error.")
            else:
                insights.append("⚠️ High average error.")
            if mape < 0.1:
                insights.append("✅ Good percentage error performance.")
            for insight in insights:
                st.sidebar.markdown(f"- {insight}")

            # Visualization
            st.subheader(f"Prediction vs Actuals vs Baseline ({model_choice})")
            baseline = np.mean(train)
            fig, ax = plt.subplots(figsize=(18, 9))
            ax.plot(df['DateTime'][len(train):], test, label='Actual')
            ax.plot(df['DateTime'][len(train):], predictions, label='Predicted')
            ax.axhline(y=baseline, color='gray', linestyle='--', label='Baseline Mean')
            ax.set_xlabel("DateTime")
            ax.set_ylabel("Power Demand (MW)")
            ax.legend()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.pyplot(fig)
                train_count = len(train)
                test_count = len(test)
                train_pct = train_count * 100 / (train_count + test_count)
                test_pct = 100 - train_pct
                st.markdown(f"**Training Data:** {train_pct:.2f}% ({train_count} points), **Predicted Data:** {test_pct:.2f}% ({test_count} points)")
            with col2:
                st.subheader("Financial Implications")
                yearly_savings_crore = yearly_savings_inr / 1e7
                if yearly_savings_inr >= 0:
                    st.markdown(f"<span style='color:green'>Average Daily Savings: ₹{daily_savings_inr:,.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:green'>Estimated Yearly Savings: ₹{yearly_savings_crore:,.2f} Crore</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:red'>Average Daily Savings: ₹{daily_savings_inr:,.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:red'>Estimated Yearly Savings: ₹{yearly_savings_crore:,.2f} Crore</span>", unsafe_allow_html=True)
                st.markdown("_Disclaimer: The average cost per MW considered is INR 4000._")
        else:
            st.error("Required columns ('DateTime', 'Year', 'Power Demand (MW)') not found in the uploaded Excel file.")
    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")
else:
    st.info("Please upload the 'Yearly Demand Profile.xlsx' file to proceed.")
