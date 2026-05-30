import pandas as pd
import numpy as np

# 1. Load preprocessed data
train_df = pd.read_csv('btc_train_processed.csv', index_col=0) # index_col=0 to load the index as well, but we won't use it for training
val_df = pd.read_csv('btc_val_processed.csv', index_col=0) # index_col=0 to load the index as well, but we won't use it for training
test_df = pd.read_csv('btc_test_processed.csv', index_col=0)

return_index = train_df.columns.get_loc('log_return') # Get the index of the 'log_return' column, which is our target variable

#2. Create sequences for LSTM
def create_sequences(data, seq_length, target_index): # seq_length is the number of time steps to look back
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_index])  # Assuming 'log_return' is the target
    return np.array(X), np.array(y)

lookback = 65 # 65 time steps (65 minutes of 5-minute data)
X_train, y_train = create_sequences(train_df.values, lookback, return_index)
X_val, y_val = create_sequences(val_df.values, lookback, return_index)
'''values is used to convert the DataFrame to a NumPy array,
 which we can use to create sequences'''



X_test, y_test = create_sequences(test_df.values, lookback, return_index)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#-----------------------------------------------------------------------------------------------------
# Save the sequences for later use in modeling
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

#------------------------------------------------------------------------------------------------------

