import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# -----------------------------

# 1. Load data
df = pd.read_csv('btc_5m_ohlcv.csv')
print(df.head())

#----------------------------
# 2. Feature engineering
# Example: Add technical indicators
df['rsi'] = ta.momentum.rsi(df['close'], window=14)

# Add log returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
print("*"*40)

# -----------------------------
#  Splitting the data
train_data = int(len(df) * 0.75)
val_data = int(len(df) * 0.15)
train_df = df[:train_data].copy() # We use .copy() to avoid SettingWithCopyWarning when we later modify train_df
val_df = df[train_data:val_data+train_data].copy() # We use .copy() to avoid SettingWithCopyWarning when we later modify val_df
test_df = df[val_data+train_data:].copy() # We use .copy() to avoid SettingWithCopyWarning when we later modify test_df
print("*"*40)
print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")
print(f"Testing data shape: {test_df.shape}")
print("*"*40)


#------------------------------

# 3. Handle missing values
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()
print("*"*40)
print(train_df.head())
#-----------------------------

# 4. Data understanding
print(train_df.describe())
print("*"*40)
print(train_df.info())
print("*"*40)
print(train_df.isnull().sum())
print("*"*40)


# -----------------------------
# 7. Data analysis
# -----------------------------
print("Training data statistics:")
print(train_df.describe())
print("*"*80)
plt.boxplot(train_df['close'], vert=True)
plt.title('Boxplot of Close Prices in Training Data')
plt.ylabel('Close Price')
plt.show()
print("Validation data statistics:")
print(val_df.describe())
print("*"*80)
plt.boxplot(val_df['close'], vert=True)
plt.title('Boxplot of Close Prices in Validation Data')
plt.ylabel('Close Price')
plt.show()
print("Testing data statistics:")
print(test_df.describe())
print("*"*80)
plt.boxplot(test_df['close'], vert=True)
plt.title('Boxplot of Close Prices in Testing Data')
plt.ylabel('Close Price')
plt.show()

# -----------------------------
# 8. Scaling and Datetime Conversion
# -----------------------------
features = ['open', 'high', 'low', 'close', 'volume', 'rsi']
scaler = MinMaxScaler()

# Important: Fit on TRAIN, transform both (prevents data leakage)
train_df[features] = scaler.fit_transform(train_df[features])
val_df[features] = scaler.transform(val_df[features])
test_df[features] = scaler.transform(test_df[features])

# Convert both to DatetimeIndex
train_df['datetime'] = pd.to_datetime(train_df['datetime'], utc=True)
val_df['datetime'] = pd.to_datetime(val_df['datetime'], utc=True)
test_df['datetime'] = pd.to_datetime(test_df['datetime'], utc=True)

train_df.set_index('datetime', inplace=True)
val_df.set_index('datetime', inplace=True)
test_df.set_index('datetime', inplace=True)

train_df.drop(columns=['timestamp'], inplace=True)
val_df.drop(columns=['timestamp'], inplace=True)
test_df.drop(columns=['timestamp'], inplace=True)

print("*"*40)

#------------------------------

# 9. Processing the date for LSTM
"""
How it works: We only take the 'day', 'hour', and 'minute' columns, as we dont need the entire
year, month, day, hour, minute, second for our LSTM model. 
We only need the day, hour, and minute to capture the time-based patterns in the data. 
The 'day' column captures the day of the month (1-31), the 'hour' column captures the hour of 
the day (0-23), and the 'minute' column captures the minute of the hour (0-59). By using these
 three columns, we can effectively represent the time information in a way that is relevant for 
 our LSTM model without including unnecessary details that may not contribute to the model's
performance.

Also, these 3 encoding (day, hour, minute) will create repeated values in the dataset and this is
fine becuase the model (LSTM) knows that the same day, hour, minute can occur multiple times and
it will learn to recognize patterns based on these time features.
"""
# Date processing in training data
train_df['hours_sin'] = np.sin(2 * np.pi * train_df.index.hour / 24)
train_df['hours_cos'] = np.cos(2 * np.pi * train_df.index.hour / 24)
train_df['week_sin'] = np.sin(2*np.pi * train_df.index.dayofweek / 7)
train_df['week_cos'] = np.cos(2*np.pi * train_df.index.dayofweek / 7)

# Date processing in validation data
val_df['hours_sin'] = np.sin(2 * np.pi * val_df.index.hour / 24)
val_df['hours_cos'] = np.cos(2 * np.pi * val_df.index.hour / 24)
val_df['week_sin'] = np.sin(2*np.pi * val_df.index.dayofweek / 7)
val_df['week_cos'] = np.cos(2*np.pi * val_df.index.dayofweek / 7)

# Date processing in testing data
test_df['hours_sin'] = np.sin(2 * np.pi * test_df.index.hour / 24)
test_df['hours_cos'] = np.cos(2 * np.pi * test_df.index.hour / 24)
test_df['week_sin'] = np.sin(2*np.pi * test_df.index.dayofweek / 7)
test_df['week_cos'] = np.cos(2*np.pi * test_df.index.dayofweek / 7)

# Checking the final processed data
print(train_df.head())
print("*"*100)
print(val_df.head())
print("*"*100)
print(test_df.head())

# Saving the data
train_df.to_csv('btc_train_processed.csv', index=True) # index=True to save the index (datetime) in the CSV file, which we will use later for creating sequences and modeling
val_df.to_csv('btc_val_processed.csv', index=True)
test_df.to_csv('btc_test_processed.csv', index=True)

# Saving the scaler for later use in inverse transformation
import joblib
joblib.dump(scaler, 'btc_scaler.pkl')

#---------------------------------------------------------------

