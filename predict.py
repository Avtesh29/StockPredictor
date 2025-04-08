import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import datetime as dt

import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler

from keras.api.layers import Dense, Dropout, LSTM
from keras.api.models import Sequential


# Configure style
plt.style.use('fivethirtyeight')

stock = "POWERGRID.NS"      # stock name
start = dt.datetime(2000, 1, 1)     # January 1st, 2001
end = dt.datetime(2024, 11, 13)     # November 13th, 2024

# Download using yahoo finance data
df = yf.download(stock, start, end)
df = df.reset_index()       # Add 'Date' column

# USEFUL FINANCE DATA FUNCTIONS
# print(df.shape)
# df.info()
# print(df.isnull().sum())
# print(df.describe())
# print(df.head())
# print(df.tail())
# print(df.columns)

# Get data into plottable format
df.to_csv("powergrid.csv")
data01 = pd.read_csv("powergrid.csv")

# Candlesticks: https://robinhood.com/us/en/learn/articles/3YzdYQ8bI4XqfnYUNj3dac/what-is-a-candlestick/
fig = go.Figure(data=[go.Candlestick(x = data01['Date'], open = data01['Open'], high = data01['High'], 
                                     low = data01['Low'], close = data01['Close'])])
# Use finance data to create a candestick plot using plotly
fig.update_layout(xaxis_rangeslider_visible=False)
# fig.show()

# Remove unneccessary data
df = df.drop('Date', axis = 1)

# Create close, open, and volume plots (can also use 'High', 'Low', etc.)
# Close Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label = f'{stock} Closing Price', linewidth = 2)
plt.title(f'{stock} Closing prices over time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.show()

# Open Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Open'], label = f'{stock} Opening Price', linewidth = 2)
plt.title(f'{stock} Opening prices over time')
plt.legend()
plt.show()

# Volume Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Volume'], label = f'{stock} Volume', linewidth = 2)
plt.title(f'{stock} Volume over time')
plt.legend()
plt.show()

# Moving Avergage: used to predict next day's stock prediciton
temp_data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print(sum(temp_data[1:6])/5)

df01 = pd.DataFrame(temp_data)
df01.rolling(5).mean()

# df['Close'] is the same as df.Close
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df.Close, label = f'{stock} Close Price', linewidth = 2)
plt.plot(ma100, label = f'{stock} Moving Average 100', linewidth = 2)
plt.plot(ma200, label = f'{stock} Moving Average 200', linewidth = 2)
plt.legend()
plt.show()

# Predict next 30 days stock; LSTM (instead of RNN)
ema100 = df.Close.ewm(span=100, adjust=False).mean()      #exponential moving average 100
ema200 = df.Close.ewm(span=200, adjust=False).mean()      #exponential moving average 200

# Exponential Moving Averages
plt.figure(figsize=(12, 6))
plt.plot(df.Close, label = f'{stock} Close Price', linewidth = 2)
plt.plot(ema100, label = f'{stock} Exp. Moving Average 100', linewidth = 2)
plt.plot(ema200, label = f'{stock} Exp. Moving Average 200', linewidth = 2)
plt.legend()
plt.show()

# Long Short-Term Memory over Recurrent Neural Network
# Training and Testing
dfsize = int(len(df))
percent70 = int(dfsize*70)
data_training = pd.DataFrame(df.Close[0: percent70 ])
data_testing = pd.DataFrame(df.Close[percent70: dfsize])

scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Model Building; LSTM -> 3d Array (batch_size, time_steps, seq_len)
# 2D -> (batch_size, units)
# 3D -> (batch_size, time_steps, units)
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)

past_100_days = data_training.tail(100)
final_df = past_100_days._append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

model.repeat()
y_predicted = model.predict(x_test)

# scaler.scale_
scaler_factor = 1 / 0.0035166
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor

# Original vs. Predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test, label = 'Original Price', linewidth = 1)
plt.plot(y_predicted, label = 'Predicted Price', linewidth = 1)
plt.legend()
plt.show()

model.save('stock_dl_model.h5')
