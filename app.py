import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import plotly.graph_objs as go

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor ")

# --------------------------------------------------------
# INPUT
# --------------------------------------------------------
stock = st.text_input("Enter Stock ID (Example: GOOG, TSLA, TCS.NS, RELIANCE.NS)", "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

google_data = yf.download(stock, start, end)

if google_data.empty:
    st.error("❌ Invalid stock symbol or no data found. Try TSLA, AAPL, TCS.NS etc.")
    st.stop()

# Load model
model = load_model("stock_prediction.keras")

st.subheader("📊 Stock Data")
st.dataframe(google_data)

splitting_len = int(len(google_data) * 0.7)

# --------------------------------------------------------
# MOVING AVERAGE PLOT (Simple Matplotlib)
# --------------------------------------------------------
st.subheader("📉 Moving Averages")

google_data['MA_250'] = google_data['Close'].rolling(250).mean()
google_data['MA_200'] = google_data['Close'].rolling(200).mean()
google_data['MA_100'] = google_data['Close'].rolling(100).mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(google_data['Close'], label="Closing Price", color='blue')
ax.plot(google_data['MA_100'], label="MA 100", color='orange')
ax.plot(google_data['MA_200'], label="MA 200", color='green')
ax.plot(google_data['MA_250'], label="MA 250", color='red')
ax.set_title("Moving Averages")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------------
# SCALING & TEST DATA
# --------------------------------------------------------
x_test = pd.DataFrame(google_data.Close[splitting_len:])
x_test.columns = ['Close']

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data, y_data = [], []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

if x_data.shape[0] == 0:
    st.error("❌ Not enough historical data to make predictions for this stock.")
    st.stop()


predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
    {
        'Original': inv_y_test.reshape(-1),
        'Predicted': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len + 100:]
)

# --------------------------------------------------------
# ORIGINAL vs PREDICTED TABLE
# --------------------------------------------------------
st.subheader("📘 Original vs Predicted – Table View")

st.dataframe(
    ploting_data.style.format({
        "Original": "{:.2f}",
        "Predicted": "{:.2f}"
    }),
    height=400,
    use_container_width=True
)


# --------------------------------------------------------
# ORIGINAL vs PREDICTED GRAPH (Interactive)
# --------------------------------------------------------
st.subheader("📈 Original vs Predicted – Graph View")

fig_pred = go.Figure()

fig_pred.add_trace(go.Scatter(
    x=ploting_data.index,
    y=ploting_data["Original"],
    mode="lines",
    name="Original",
    line=dict(width=2)
))

fig_pred.add_trace(go.Scatter(
    x=ploting_data.index,
    y=ploting_data["Predicted"],
    mode="lines",
    name="Predicted",
    line=dict(width=2)
))

fig_pred.update_layout(
    template="plotly_white",
    height=500,
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_pred, use_container_width=True)



# --------------------------------------------------------
# NEXT DAY PREDICTION
# --------------------------------------------------------
st.subheader("🔮 Next Day Prediction & Recommendation")

last_100 = google_data.Close[-100:].values.reshape(-1, 1)
scaled_last_100 = scaler.transform(last_100)
x_input = scaled_last_100.reshape(1, 100, 1)

next_day_scaled = model.predict(x_input)
next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]

# Detect currency
stock_upper = stock.upper()
currency_symbol = "₹" if stock_upper.endswith((".NS", ".BO", ".BSE", ".NSE")) else "$"

# Current price safely retrieved
current_price = google_data.Close.iloc[-1]

percent_change = float(((next_day_price - current_price) / current_price) * 100)

st.write(f"📌 Predicted Next Day Close: {currency_symbol}{next_day_price:.2f}")
st.write(f"📈 Expected Change: {percent_change:.2f}%")

st.subheader("💡 Recommendation")

if percent_change > 3:
    st.success("🟢 **Strong Buy** — Good upward momentum expected.")
elif percent_change > 1:
    st.info("🟡 **Buy / Hold** — Mild positive trend.")
elif -1 <= percent_change <= 1:
    st.warning("⚪ **Neutral** — Market may stay sideways.")
elif percent_change < -1:
    st.error("🔴 **Not a good buy** — Expected to fall.")
else:
    st.write("⚠️ Market uncertain.")


# --------------------------------------------------------
# 7-DAY FORECAST
# --------------------------------------------------------
st.subheader("📅 7-Day Future Forecast")

# Use last 100 days from full Close
full_close = google_data.Close.values.reshape(-1, 1)
full_scaled = scaler.transform(full_close)

last_seq = full_scaled[-100:].reshape(100, 1)

future_scaled = []
seq_buffer = last_seq.copy()

for _ in range(7):  
    x_in = seq_buffer.reshape(1, 100, 1)
    pred_scaled = model.predict(x_in)
    
    # Save prediction
    future_scaled.append(pred_scaled[0][0])

    # Update buffer (drop first row, append prediction)
    seq_buffer = np.vstack([seq_buffer[1:], [[pred_scaled[0][0]]]])

# Inverse transform predictions
future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).reshape(-1)

# Create date index
future_dates = pd.bdate_range(start=google_data.index[-1] + pd.Timedelta(days=1), periods=7)

forecast_df = pd.DataFrame({
    "Predicted_Price": future_prices
}, index=future_dates)

st.dataframe(forecast_df.style.format({"Predicted_Price": "{:.2f}"}), height=300)

# Plot 7-day forecast graph
fig_7 = go.Figure()
fig_7.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted_Price"], 
                           mode="lines+markers", name="7-Day Forecast"))

# Add last 30 days for comparison
fig_7.add_trace(go.Scatter(
    x=google_data.index[-30:], 
    y=google_data["Close"].iloc[-30:], 
    mode="lines",
    name="Recent Actual"
))

fig_7.update_layout(
    template="plotly_white",
    height=450,
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified"
)

st.plotly_chart(fig_7, use_container_width=True)


