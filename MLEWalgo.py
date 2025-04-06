import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import yfinance as yf
import talib
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 5000)

# Control variables
days = 365
interval = "1D"
Ticker = 'MSTR'

# Specify start and end dates
end_date = date.today()
start_date = end_date - timedelta(days=days)
print('Start Date:', start_date)
print('End Date:', end_date)

# Download historical data
df = yf.download(Ticker, start=start_date, end=end_date, interval=interval)
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Calculate moving averages
df['ma20'] = df['Close'].rolling(20).mean()
df['ma50'] = df['Close'].rolling(50).mean()

# Calculate Bollinger Bands
df['upper_band'] = df['ma20'] + (df['Close'].rolling(window=20).std() * 2)
df['lower_band'] = df['ma20'] - (df['Close'].rolling(window=20).std() * 2)

# Calculate RSI and MACD
df['rsi14'] = talib.RSI(df.Close, 14)
df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=5)

# Simplified Elliott Wave identification
def identify_elliott_waves(df):
    df['elliott_wave_signal'] = ''
    for i in range(2, len(df)-2):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1] and df['Close'].iloc[i] > df['Close'].iloc[i+1]:
            df['elliott_wave_signal'].iloc[i] = 'S'  # Potential peak (sell signal)
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1] and df['Close'].iloc[i] < df['Close'].iloc[i+1]:
            df['elliott_wave_signal'].iloc[i] = 'B'  # Potential trough (buy signal)

identify_elliott_waves(df)

# Calculate RSV (Raw Stochastic Value)
def calculate_rsv(df, period=14):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    rsv = ((df['Close'] - low_min) / (high_max - low_min)) * 100
    return rsv

df['rsv'] = calculate_rsv(df)

# Calculate KDJ
def calculate_kdj(df, period=14):
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min)).rolling(window=3).mean()
    d = k.rolling(window=3).mean()
    j = 3 * k - 2 * d
    return k, d, j

df['k'], df['d'], df['j'] = calculate_kdj(df)

# Initialize signals
df['signal_macd'] = ''
df['signal_rsv'] = ''
df['signal_kdj'] = ''
df['signal_bollinger'] = ''
df['final_signal'] = ''
df['macd_bullish_div'] = 0
df['macd_bearish_div'] = 0

# Generate signals based on MACD
df.loc[df['macdsignal'] <= df['macdhist'], 'signal_macd'] = 'S'  # MACD bearish
df.loc[df['macdsignal'] >= df['macdhist'], 'signal_macd'] = 'B'  # MACD bullish

# Generate signals based on RSV
df.loc[df['rsv'] >= 80, 'signal_rsv'] = 'S'  # Overbought
df.loc[df['rsv'] <= 20, 'signal_rsv'] = 'B'  # Oversold

# Generate signals based on KDJ
df.loc[df['k'] > 80, 'signal_kdj'] = 'S'  # KDJ overbought
df.loc[df['k'] < 20, 'signal_kdj'] = 'B'  # KDJ oversold

# Generate signals based on Bollinger Bands
df['signal_bollinger'] = np.where(df['Close'] > df['upper_band'], 'S',
                                   np.where(df['Close'] < df['lower_band'], 'B', ''))

# Identify MACD Divergence
for i in range(1, len(df) - 1):
    if (df['Close'].iloc[i] < df['Close'].iloc[i-1] and 
        df['Close'].iloc[i] < df['Close'].iloc[i+1] and
        df['macd'].iloc[i] > df['macd'].iloc[i-1] and 
        df['macd'].iloc[i] > df['macd'].iloc[i+1]):
            df['macd_bullish_div'].iloc[i] = 1  # Mark bullish divergence

    if (df['Close'].iloc[i] > df['Close'].iloc[i-1] and 
        df['Close'].iloc[i] > df['Close'].iloc[i+1] and
        df['macd'].iloc[i] < df['macd'].iloc[i-1] and 
        df['macd'].iloc[i] < df['macd'].iloc[i+1]):
            df['macd_bearish_div'].iloc[i] = 1  # Mark bearish divergence

# Count the signals for Buy and Sell
buy_signals = (df['signal_rsv'] == 'B').astype(int) + \
              (df['signal_macd'] == 'B').astype(int) + \
              (df['signal_kdj'] == 'B').astype(int) + \
              (df['signal_bollinger'] == 'B').astype(int) + \
              df['macd_bullish_div']

sell_signals = (df['signal_rsv'] == 'S').astype(int) + \
               (df['signal_macd'] == 'S').astype(int) + \
               (df['signal_kdj'] == 'S').astype(int) + \
               (df['signal_bollinger'] == 'S').astype(int) + \
               df['macd_bearish_div']

# Assign final signals based on combined logic
df['final_signal'] = np.where(buy_signals >= 4, 'B',
                              np.where(sell_signals >= 4, 'S', ''))

# ARIMA Forecasting
model = ARIMA(df['Close'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=1)
    
if not forecast.empty:
    forecast_value = forecast.iloc[0]
    print(f"Forecast value: {forecast_value}")
else:
    print("Forecast is empty.")
        
# LSTM Forecasting
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=100, batch_size=32)

last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
predicted_price = model.predict(last_data)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

# Combine forecasts for signals
if predicted_price > df['Close'].iloc[-1]:
    df['final_signal'].iloc[-1] = 'B'
elif predicted_price < df['Close'].iloc[-1]:
    df['final_signal'].iloc[-1] = 'S'

# Plotting the stock price and combined signals
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['ma20'], label='20-Day MA', alpha=0.75)
plt.plot(df['ma50'], label='50-Day MA', alpha=0.75)
plt.plot(df['upper_band'], label='Upper Bollinger Band', linestyle='--', alpha=0.5)
plt.plot(df['lower_band'], label='Lower Bollinger Band', linestyle='--', alpha=0.5)

plt.scatter(df.index[df['final_signal'] == 'B'], df[df['final_signal'] == 'B']['Close'], 
            marker='^', color='g', label='Buy Signal', alpha=1)

plt.scatter(df.index[df['final_signal'] == 'S'], df[df['final_signal'] == 'S']['Close'], 
            marker='v', color='r', label='Sell Signal', alpha=1)

plt.title(f'{Ticker} Price with Combined Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Save DataFrame to Excel
df.to_excel(f'{Ticker}_with_signals.xlsx')