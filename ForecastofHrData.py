import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import torch
import torch.nn as nn

st.set_page_config(layout="wide")
st.title("🔌 Hourly Power Demand Forecasting for India")

uploaded_file = st.file_uploader("📤 Upload Hourly Power Demand Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine='openpyxl')

    # Show column names for debugging
    st.write("Uploaded file columns:", df.columns.tolist())

    # Validate required columns
    required_cols = ['Country', 'Year', 'DateTime', 'Power Demand (MW)']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Required columns missing. Expected columns: {required_cols}")
        st.stop()

    
    df['Datetime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df = df[df['Country'] == 'India'].sort_values(by='Datetime')

    model_list = [
        "SARIMAX", "RandomForest", "LinearRegression", "SVR", "XGBoost", "LSTM", "GRU", "Hybrid"
    ]

    if "model_selector" not in st.session_state:
        st.session_state.model_selector = model_list[0]

    st.sidebar.header("⚙️ Model Configuration")
    selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_list, index=model_list.index(st.session_state.model_selector))
    st.sidebar.subheader("📊 Accuracy Metrics")

    series = df['Power Demand (MW)'].values
    dates = df['Datetime'].values
    train, test = series[:int(0.7*len(series))], series[int(0.7*len(series)):]

    def create_features(data, window=24):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def train_model(model_name, X_train, y_train, X_test, scaler, train, test):
        if model_name == "SARIMAX":
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(test))
        elif model_name in ["RandomForest", "LinearRegression", "SVR", "XGBoost"]:
            if model_name == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "LinearRegression":
                model = LinearRegression()
            elif model_name == "SVR":
                model = SVR(kernel='rbf')
            elif model_name == "XGBoost":
                model = XGBRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            forecast_scaled = model.predict(X_test)
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        elif model_name in ["LSTM", "GRU", "Hybrid"]:
            X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
            y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
            X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

            class TimeSeriesModel(nn.Module):
                def __init__(self, model_type):
                    super().__init__()
                    if model_type == "LSTM":
                        self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                    elif model_type == "GRU":
                        self.rnn = nn.GRU(input_size=1, hidden_size=50, batch_first=True)
                    elif model_type == "Hybrid":
                        self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
                        self.fc1 = nn.Linear(50, 25)
                        self.fc2 = nn.Linear(25, 1)
                    else:
                        raise ValueError("Invalid model type")

                    if model_type != "Hybrid":
                        self.fc = nn.Linear(50, 1)

                    self.model_type = model_type

                def forward(self, x):
                    out, _ = self.rnn(x)
                    out = out[:, -1, :]
                    if self.model_type == "Hybrid":
                        out = torch.relu(self.fc1(out))
                        out = self.fc2(out)
                    else:
                        out = self.fc(out)
                    return out

            model = TimeSeriesModel(model_name)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                output = model(X_train_torch)
                loss = criterion(output, y_train_torch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                forecast_scaled = model(X_test_torch).squeeze().numpy()

            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
            test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        return forecast, test

    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    window = 24
    X_train, y_train = create_features(scaled_series[:int(0.7*len(series))], window)
    X_test, y_test = create_features(scaled_series[int(0.7*len(series))-window:], window)

    forecast, test = train_model(selected_model, X_train, y_train, X_test, scaler, train, test)

    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    r2_raw = r2_score(test, forecast)
    r2 = max(0.0, r2_raw)

    st.sidebar.write(f"**R² Score**: {r2:.2f}")
    st.sidebar.write(f"**RMSE**: {rmse:.2f}")
    st.sidebar.write(f"**MAE**: {mae:.2f}")

    st.subheader(f"📈 Forecast vs Actual using {selected_model}")
    plot_df = pd.DataFrame({
        'Datetime': dates[len(dates) - len(test):],
        'Actual': test,
        'Predicted': forecast
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x='Datetime', y='Actual', label='Actual', ax=ax)
    sns.lineplot(data=plot_df, x='Datetime', y='Predicted', label='Predicted', ax=ax)
    ax.set_ylabel("Power Demand (MW)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
