{
    "chunks": [
        {
            "type": "txt",
            "chunk_number": 1,
            "lines": [
                {
                    "line_number": 1,
                    "text": ""
                },
                {
                    "line_number": 2,
                    "text": "import streamlit as st"
                },
                {
                    "line_number": 3,
                    "text": "import pandas as pd"
                },
                {
                    "line_number": 4,
                    "text": "import numpy as np"
                },
                {
                    "line_number": 5,
                    "text": "import seaborn as sns"
                },
                {
                    "line_number": 6,
                    "text": "import matplotlib.pyplot as plt"
                },
                {
                    "line_number": 7,
                    "text": "from sklearn.ensemble import RandomForestRegressor"
                },
                {
                    "line_number": 8,
                    "text": "from sklearn.linear_model import LinearRegression"
                },
                {
                    "line_number": 9,
                    "text": "from sklearn.svm import SVR"
                },
                {
                    "line_number": 10,
                    "text": "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score"
                },
                {
                    "line_number": 11,
                    "text": "from sklearn.preprocessing import MinMaxScaler"
                },
                {
                    "line_number": 12,
                    "text": "from statsmodels.tsa.statespace.sarimax import SARIMAX"
                },
                {
                    "line_number": 13,
                    "text": "from xgboost import XGBRegressor"
                },
                {
                    "line_number": 14,
                    "text": "import torch"
                },
                {
                    "line_number": 15,
                    "text": "import torch.nn as nn"
                },
                {
                    "line_number": 16,
                    "text": ""
                },
                {
                    "line_number": 17,
                    "text": "st.set_page_config(layout=\"wide\")"
                },
                {
                    "line_number": 18,
                    "text": "st.title(\"\ud83d\udd2e Short-Term Intra-Day Forecast of Power Demand\")"
                },
                {
                    "line_number": 19,
                    "text": ""
                },
                {
                    "line_number": 20,
                    "text": "uploaded_file = st.file_uploader(\"\ud83d\udce4 Upload Power Demand Excel File\", type=[\"xlsx\"])"
                },
                {
                    "line_number": 21,
                    "text": ""
                },
                {
                    "line_number": 22,
                    "text": "if uploaded_file is not None:"
                },
                {
                    "line_number": 23,
                    "text": "df = pd.read_excel(uploaded_file, engine='openpyxl')"
                },
                {
                    "line_number": 24,
                    "text": "df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))"
                },
                {
                    "line_number": 25,
                    "text": "df.sort_values(by=['State', 'Datetime'], inplace=True)"
                },
                {
                    "line_number": 26,
                    "text": ""
                },
                {
                    "line_number": 27,
                    "text": "mw_rates = {"
                },
                {
                    "line_number": 28,
                    "text": "'Maharashtra': 5.2, 'Tamil Nadu': 4.8, 'Karnataka': 5.0, 'Gujarat': 5.1,"
                },
                {
                    "line_number": 29,
                    "text": "'West Bengal': 4.9, 'Rajasthan': 5.3, 'Uttar Pradesh': 4.7,"
                },
                {
                    "line_number": 30,
                    "text": "'Kerala': 5.4, 'Punjab': 5.0, 'Bihar': 4.6"
                },
                {
                    "line_number": 31,
                    "text": "}"
                },
                {
                    "line_number": 32,
                    "text": ""
                },
                {
                    "line_number": 33,
                    "text": "model_list = ["
                },
                {
                    "line_number": 34,
                    "text": "\"SARIMAX\", \"RandomForest\", \"LinearRegression\", \"SVR\", \"XGBoost\", \"LSTM\", \"GRU\", \"Hybrid\""
                },
                {
                    "line_number": 35,
                    "text": "]"
                },
                {
                    "line_number": 36,
                    "text": ""
                },
                {
                    "line_number": 37,
                    "text": "if \"model_selector\" not in st.session_state:"
                },
                {
                    "line_number": 38,
                    "text": "st.session_state.model_selector = model_list[0]"
                },
                {
                    "line_number": 39,
                    "text": ""
                },
                {
                    "line_number": 40,
                    "text": "st.sidebar.header(\"\u2699\ufe0f Model Configuration\")"
                },
                {
                    "line_number": 41,
                    "text": "selected_model = st.sidebar.selectbox(\"Choose Forecasting Model\", model_list, index=model_list.index(st.session_state.model_selector))"
                },
                {
                    "line_number": 42,
                    "text": "st.sidebar.subheader(\"\ud83d\udcca Accuracy Metrics\")"
                },
                {
                    "line_number": 43,
                    "text": ""
                },
                {
                    "line_number": 44,
                    "text": "state = st.selectbox(\"\ud83d\udccd Select State\", df['State'].unique())"
                },
                {
                    "line_number": 45,
                    "text": "rate = mw_rates.get(state, 5.0)"
                },
                {
                    "line_number": 46,
                    "text": ""
                },
                {
                    "line_number": 47,
                    "text": "state_df = df[df['State'] == state].sort_values('Datetime')"
                },
                {
                    "line_number": 48,
                    "text": "series = state_df['Power Demand (MW)'].values[:100]"
                },
                {
                    "line_number": 49,
                    "text": "dates = state_df['Datetime'].values[:100]"
                },
                {
                    "line_number": 50,
                    "text": "train, test = series[:70], series[70:]"
                },
                {
                    "line_number": 51,
                    "text": ""
                },
                {
                    "line_number": 52,
                    "text": "def create_features(data, window=5):"
                },
                {
                    "line_number": 53,
                    "text": "X, y = [], []"
                },
                {
                    "line_number": 54,
                    "text": "for i in range(window, len(data)):"
                },
                {
                    "line_number": 55,
                    "text": "X.append(data[i-window:i])"
                },
                {
                    "line_number": 56,
                    "text": "y.append(data[i])"
                },
                {
                    "line_number": 57,
                    "text": "return np.array(X), np.array(y)"
                },
                {
                    "line_number": 58,
                    "text": ""
                },
                {
                    "line_number": 59,
                    "text": "def train_model(model_name, X_train, y_train, X_test, scaler, train, test):"
                },
                {
                    "line_number": 60,
                    "text": "if model_name == \"SARIMAX\":"
                },
                {
                    "line_number": 61,
                    "text": "model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))"
                },
                {
                    "line_number": 62,
                    "text": "model_fit = model.fit(disp=False)"
                },
                {
                    "line_number": 63,
                    "text": "forecast = model_fit.forecast(steps=30)"
                },
                {
                    "line_number": 64,
                    "text": "elif model_name in [\"RandomForest\", \"LinearRegression\", \"SVR\", \"XGBoost\"]:"
                },
                {
                    "line_number": 65,
                    "text": "if model_name == \"RandomForest\":"
                },
                {
                    "line_number": 66,
                    "text": "model = RandomForestRegressor(n_estimators=100, random_state=42)"
                },
                {
                    "line_number": 67,
                    "text": "elif model_name == \"LinearRegression\":"
                },
                {
                    "line_number": 68,
                    "text": "model = LinearRegression()"
                },
                {
                    "line_number": 69,
                    "text": "elif model_name == \"SVR\":"
                },
                {
                    "line_number": 70,
                    "text": "model = SVR(kernel='rbf')"
                },
                {
                    "line_number": 71,
                    "text": "elif model_name == \"XGBoost\":"
                },
                {
                    "line_number": 72,
                    "text": "model = XGBRegressor(n_estimators=100, random_state=42)"
                },
                {
                    "line_number": 73,
                    "text": ""
                },
                {
                    "line_number": 74,
                    "text": "model.fit(X_train, y_train)"
                },
                {
                    "line_number": 75,
                    "text": "forecast_scaled = model.predict(X_test)"
                },
                {
                    "line_number": 76,
                    "text": "forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()"
                },
                {
                    "line_number": 77,
                    "text": "test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()"
                },
                {
                    "line_number": 78,
                    "text": "elif model_name in [\"LSTM\", \"GRU\", \"Hybrid\"]:"
                },
                {
                    "line_number": 79,
                    "text": "X_train_torch = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)"
                },
                {
                    "line_number": 80,
                    "text": "y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)"
                },
                {
                    "line_number": 81,
                    "text": "X_test_torch = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)"
                },
                {
                    "line_number": 82,
                    "text": ""
                },
                {
                    "line_number": 83,
                    "text": "class TimeSeriesModel(nn.Module):"
                },
                {
                    "line_number": 84,
                    "text": "def __init__(self, model_type):"
                },
                {
                    "line_number": 85,
                    "text": "super().__init__()"
                },
                {
                    "line_number": 86,
                    "text": "if model_type == \"LSTM\":"
                },
                {
                    "line_number": 87,
                    "text": "self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)"
                },
                {
                    "line_number": 88,
                    "text": "elif model_type == \"GRU\":"
                },
                {
                    "line_number": 89,
                    "text": "self.rnn = nn.GRU(input_size=1, hidden_size=50, batch_first=True)"
                },
                {
                    "line_number": 90,
                    "text": "elif model_type == \"Hybrid\":"
                },
                {
                    "line_number": 91,
                    "text": "self.rnn = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)"
                },
                {
                    "line_number": 92,
                    "text": "self.fc1 = nn.Linear(50, 25)"
                },
                {
                    "line_number": 93,
                    "text": "self.fc2 = nn.Linear(25, 1)"
                },
                {
                    "line_number": 94,
                    "text": "else:"
                },
                {
                    "line_number": 95,
                    "text": "raise ValueError(\"Invalid model type\")"
                },
                {
                    "line_number": 96,
                    "text": ""
                },
                {
                    "line_number": 97,
                    "text": "if model_type != \"Hybrid\":"
                },
                {
                    "line_number": 98,
                    "text": "self.fc = nn.Linear(50, 1)"
                },
                {
                    "line_number": 99,
                    "text": ""
                },
                {
                    "line_number": 100,
                    "text": "self.model_type = model_type"
                },
                {
                    "line_number": 101,
                    "text": ""
                },
                {
                    "line_number": 102,
                    "text": "def forward(self, x):"
                },
                {
                    "line_number": 103,
                    "text": "out, _ = self.rnn(x)"
                },
                {
                    "line_number": 104,
                    "text": "out = out[:, -1, :]"
                },
                {
                    "line_number": 105,
                    "text": "if self.model_type == \"Hybrid\":"
                },
                {
                    "line_number": 106,
                    "text": "out = torch.relu(self.fc1(out))"
                },
                {
                    "line_number": 107,
                    "text": "out = self.fc2(out)"
                },
                {
                    "line_number": 108,
                    "text": "else:"
                },
                {
                    "line_number": 109,
                    "text": "out = self.fc(out)"
                },
                {
                    "line_number": 110,
                    "text": "return out"
                },
                {
                    "line_number": 111,
                    "text": ""
                },
                {
                    "line_number": 112,
                    "text": "model = TimeSeriesModel(model_name)"
                },
                {
                    "line_number": 113,
                    "text": "criterion = nn.MSELoss()"
                },
                {
                    "line_number": 114,
                    "text": "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)"
                },
                {
                    "line_number": 115,
                    "text": ""
                },
                {
                    "line_number": 116,
                    "text": "for epoch in range(50):"
                },
                {
                    "line_number": 117,
                    "text": "model.train()"
                },
                {
                    "line_number": 118,
                    "text": "optimizer.zero_grad()"
                },
                {
                    "line_number": 119,
                    "text": "output = model(X_train_torch)"
                },
                {
                    "line_number": 120,
                    "text": "loss = criterion(output, y_train_torch)"
                },
                {
                    "line_number": 121,
                    "text": "loss.backward()"
                },
                {
                    "line_number": 122,
                    "text": "optimizer.step()"
                },
                {
                    "line_number": 123,
                    "text": ""
                },
                {
                    "line_number": 124,
                    "text": "model.eval()"
                },
                {
                    "line_number": 125,
                    "text": "with torch.no_grad():"
                },
                {
                    "line_number": 126,
                    "text": "forecast_scaled = model(X_test_torch).squeeze().numpy()"
                },
                {
                    "line_number": 127,
                    "text": ""
                },
                {
                    "line_number": 128,
                    "text": "forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()"
                },
                {
                    "line_number": 129,
                    "text": "test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()"
                },
                {
                    "line_number": 130,
                    "text": ""
                },
                {
                    "line_number": 131,
                    "text": "baseline = np.full_like(test, np.mean(train))"
                },
                {
                    "line_number": 132,
                    "text": "mw_savings = np.sum(baseline - forecast)"
                },
                {
                    "line_number": 133,
                    "text": "financial_gain = mw_savings * rate"
                },
                {
                    "line_number": 134,
                    "text": "yearly_gain = financial_gain * 365"
                },
                {
                    "line_number": 135,
                    "text": "return forecast, test, financial_gain, yearly_gain"
                },
                {
                    "line_number": 136,
                    "text": ""
                },
                {
                    "line_number": 137,
                    "text": "scaler = MinMaxScaler()"
                },
                {
                    "line_number": 138,
                    "text": "scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()"
                },
                {
                    "line_number": 139,
                    "text": "window = 5"
                },
                {
                    "line_number": 140,
                    "text": "X_train, y_train = create_features(scaled_series[:70], window)"
                },
                {
                    "line_number": 141,
                    "text": "X_test, y_test = create_features(scaled_series[70-window:100], window)"
                },
                {
                    "line_number": 142,
                    "text": ""
                },
                {
                    "line_number": 143,
                    "text": "forecast, test, financial_gain, yearly_gain = train_model("
                },
                {
                    "line_number": 144,
                    "text": "selected_model, X_train, y_train, X_test, scaler, train, test"
                },
                {
                    "line_number": 145,
                    "text": ")"
                },
                {
                    "line_number": 146,
                    "text": ""
                },
                {
                    "line_number": 147,
                    "text": "rmse = np.sqrt(mean_squared_error(test, forecast))"
                },
                {
                    "line_number": 148,
                    "text": "mae = mean_absolute_error(test, forecast)"
                },
                {
                    "line_number": 149,
                    "text": "r2_raw = r2_score(test, forecast)"
                },
                {
                    "line_number": 150,
                    "text": "r2 = max(0.0, r2_raw)"
                },
                {
                    "line_number": 151,
                    "text": ""
                },
                {
                    "line_number": 152,
                    "text": "st.sidebar.write(f\"**R\u00b2 Score**: {r2:.2f}\")"
                },
                {
                    "line_number": 153,
                    "text": "st.sidebar.write(f\"**RMSE**: {rmse:.2f}\")"
                },
                {
                    "line_number": 154,
                    "text": "st.sidebar.write(f\"**MAE**: {mae:.2f}\")"
                },
                {
                    "line_number": 155,
                    "text": ""
                },
                {
                    "line_number": 156,
                    "text": "st.sidebar.subheader(\"\ud83d\udca1 Model Insights\")"
                },
                {
                    "line_number": 157,
                    "text": "insights = []"
                },
                {
                    "line_number": 158,
                    "text": "if r2_raw > 0.85 and rmse < 100 and mae < 100:"
                },
                {
                    "line_number": 159,
                    "text": "st.sidebar.success(\"\u2705 Recommended Model\")"
                },
                {
                    "line_number": 160,
                    "text": "insights = ["
                },
                {
                    "line_number": 161,
                    "text": "\"- High accuracy and low error.\","
                },
                {
                    "line_number": 162,
                    "text": "\"- Suitable for short-term forecasting.\","
                },
                {
                    "line_number": 163,
                    "text": "\"- Reliable for operational planning.\","
                },
                {
                    "line_number": 164,
                    "text": "\"- Captures demand patterns effectively.\","
                },
                {
                    "line_number": 165,
                    "text": "\"- Minimal deviation from actual values.\""
                },
                {
                    "line_number": 166,
                    "text": "]"
                },
                {
                    "line_number": 167,
                    "text": "elif r2_raw > 0.7:"
                },
                {
                    "line_number": 168,
                    "text": "st.sidebar.warning(\"\u26a0\ufe0f Moderate Accuracy\")"
                },
                {
                    "line_number": 169,
                    "text": "insights = ["
                },
                {
                    "line_number": 170,
                    "text": "\"- Acceptable performance.\","
                },
                {
                    "line_number": 171,
                    "text": "\"- May benefit from tuning or more data.\","
                },
                {
                    "line_number": 172,
                    "text": "\"- Consider ensemble or hybrid approaches.\","
                },
                {
                    "line_number": 173,
                    "text": "\"- Captures general trends but may miss spikes.\","
                },
                {
                    "line_number": 174,
                    "text": "\"- Useful for preliminary planning.\""
                },
                {
                    "line_number": 175,
                    "text": "]"
                },
                {
                    "line_number": 176,
                    "text": "else:"
                },
                {
                    "line_number": 177,
                    "text": "st.sidebar.error(\"\u274c Low Accuracy\")"
                },
                {
                    "line_number": 178,
                    "text": "insights = ["
                },
                {
                    "line_number": 179,
                    "text": "\"- High error and low correlation.\","
                },
                {
                    "line_number": 180,
                    "text": "\"- May not capture demand patterns well.\","
                },
                {
                    "line_number": 181,
                    "text": "\"- Consider alternative models or preprocessing.\","
                },
                {
                    "line_number": 182,
                    "text": "\"- Not suitable for critical forecasting.\","
                },
                {
                    "line_number": 183,
                    "text": "\"- Requires significant improvement.\""
                },
                {
                    "line_number": 184,
                    "text": "]"
                },
                {
                    "line_number": 185,
                    "text": "for point in insights:"
                },
                {
                    "line_number": 186,
                    "text": "st.sidebar.markdown(f\"- {point}\")"
                },
                {
                    "line_number": 187,
                    "text": ""
                },
                {
                    "line_number": 188,
                    "text": "baseline = np.full_like(test, np.mean(train))"
                },
                {
                    "line_number": 189,
                    "text": "mw_savings = np.sum(baseline - forecast)"
                },
                {
                    "line_number": 190,
                    "text": ""
                },
                {
                    "line_number": 191,
                    "text": "col1, col2 = st.columns([3, 1])"
                },
                {
                    "line_number": 192,
                    "text": ""
                },
                {
                    "line_number": 193,
                    "text": "with col1:"
                },
                {
                    "line_number": 194,
                    "text": "st.subheader(f\"\ud83d\udcc8 Forecast vs Actual using {selected_model}\")"
                },
                {
                    "line_number": 195,
                    "text": "plot_df = pd.DataFrame({"
                },
                {
                    "line_number": 196,
                    "text": "'Datetime': dates[100 - len(test):100],"
                },
                {
                    "line_number": 197,
                    "text": "'Actual': test,"
                },
                {
                    "line_number": 198,
                    "text": "'Baseline': baseline,"
                },
                {
                    "line_number": 199,
                    "text": "'Predicted': forecast"
                },
                {
                    "line_number": 200,
                    "text": "})"
                },
                {
                    "line_number": 201,
                    "text": ""
                },
                {
                    "line_number": 202,
                    "text": "fig, ax = plt.subplots(figsize=(12, 6))"
                },
                {
                    "line_number": 203,
                    "text": "sns.lineplot(data=plot_df, x='Datetime', y='Actual', label='Actual', ax=ax)"
                },
                {
                    "line_number": 204,
                    "text": "sns.lineplot(data=plot_df, x='Datetime', y='Baseline', label='Baseline', ax=ax)"
                },
                {
                    "line_number": 205,
                    "text": "sns.lineplot(data=plot_df, x='Datetime', y='Predicted', label='Predicted', ax=ax)"
                },
                {
                    "line_number": 206,
                    "text": "ax.set_ylabel(\"Power Demand (MW)\")"
                },
                {
                    "line_number": 207,
                    "text": "plt.xticks(rotation=45)"
                },
                {
                    "line_number": 208,
                    "text": "st.pyplot(fig)"
                },
                {
                    "line_number": 209,
                    "text": "st.caption(\"\ud83d\udccc Disclaimer: Model trained on 70 blocks of 15-minute data and predicted for the last 30 blocks.\")"
                },
                {
                    "line_number": 210,
                    "text": ""
                },
                {
                    "line_number": 211,
                    "text": "with col2:"
                },
                {
                    "line_number": 212,
                    "text": "st.subheader(\"\ud83d\udcb0 Financial Highlights\")"
                },
                {
                    "line_number": 213,
                    "text": "st.markdown(f\"<h5><strong>MW Savings:</strong> {mw_savings:.2f} MW</h5>\", unsafe_allow_html=True)"
                },
                {
                    "line_number": 214,
                    "text": "st.markdown(f\"<h5><strong>Daily Financial Gain:</strong> \u20b9{financial_gain:,.2f}</h5>\", unsafe_allow_html=True)"
                },
                {
                    "line_number": 215,
                    "text": "st.markdown(f\"<h5><strong>Estimated Yearly Gain:</strong> \u20b9{yearly_gain:,.2f}</h5>\", unsafe_allow_html=True)"
                },
                {
                    "line_number": 216,
                    "text": ""
                },
                {
                    "line_number": 217,
                    "text": "if st.button(\"\ud83d\udd0d Optimize\", help=\"Click to select the model with highest financial gain\"):"
                },
                {
                    "line_number": 218,
                    "text": "best_model = None"
                },
                {
                    "line_number": 219,
                    "text": "best_gain = -np.inf"
                },
                {
                    "line_number": 220,
                    "text": "for model_name in model_list:"
                },
                {
                    "line_number": 221,
                    "text": "try:"
                },
                {
                    "line_number": 222,
                    "text": "forecast_opt, test_opt, gain_opt, _ = train_model("
                },
                {
                    "line_number": 223,
                    "text": "model_name, X_train, y_train, X_test, scaler, train, test"
                },
                {
                    "line_number": 224,
                    "text": ")"
                },
                {
                    "line_number": 225,
                    "text": "if gain_opt > best_gain:"
                },
                {
                    "line_number": 226,
                    "text": "best_gain = gain_opt"
                },
                {
                    "line_number": 227,
                    "text": "best_model = model_name"
                },
                {
                    "line_number": 228,
                    "text": "except Exception:"
                },
                {
                    "line_number": 229,
                    "text": "continue"
                },
                {
                    "line_number": 230,
                    "text": "if best_model:"
                },
                {
                    "line_number": 231,
                    "text": "st.session_state.model_selector = best_model"
                },
                {
                    "line_number": 232,
                    "text": "st.success(f\"\u2705 Optimized Model Selected: {best_model}\")"
                },
                {
                    "line_number": 233,
                    "text": "st.markdown(f\"\ud83d\udcb0 Highest Daily Financial Gain: \u20b9{best_gain:,.2f}\")"
                },
                {
                    "line_number": 234,
                    "text": "else:"
                },
                {
                    "line_number": 235,
                    "text": "st.error(\"\u274c Optimization failed. Please check your data or model configurations.\")"
                },
                {
                    "line_number": 236,
                    "text": ""
                },
                {
                    "line_number": 237,
                    "text": "st.caption(\"\ud83d\udca1 Optimize your model for power demand as per the highest financial savings\")"
                }
            ],
            "token_count": 2084
        }
    ]
}
