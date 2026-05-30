import numpy as np
import torch
import copy
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import joblib
import pandas as pd

# 1. Load sequences
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# 2. Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) # float32 is the standard data type for neural network inputs in PyTorch, it allows for efficient computation while maintaining sufficient precision for training. 
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
print(f"X_train_tensor shape: {X_train_tensor.shape}, y_train_tensor shape: {y_train_tensor.shape}")
print(f"X_val_tensor shape: {X_val_tensor.shape}, y_val_tensor shape: {y_val_tensor.shape}")
print(f"X_test_tensor shape: {X_test_tensor.shape}, y_test_tensor shape: {y_test_tensor.shape}")


# 3. Build LSTM model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel,self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(0.1)
 
        self.fc = nn.Linear(hidden_size, 1) 

    def forward(self, x):   
        lstm_out, _ = self.lstm(x) # '_' is the hidden and cell state, which is already captured in lstm_out, we don't need it separately for our prediction task.
        # lstm_out shape: (batch_size, seq_length, hidden_size), this is the output of the LSTM layer for each time step in the sequence

        lstm_out = self.dropout(lstm_out) # apply dropout to the LSTM output to prevent overfitting

        out = self.fc(lstm_out[:, -1, :]) # out shape: (batch_size, 1), we take the output of the last time step (lstm_out[:, -1, :]) and pass it through the fully connected layer to get the final prediction

        return out


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
input_size = X_train_tensor.shape[2] # number of features in the input data, 10 in this case, since we have 10 features in our input sequences
print(f"Input size for LSTM: {input_size}")
hidden_size = 64 # number of features in the hidden state of the LSTM, this is a hyperparameter that can be tuned based on the complexity of the data and the model's performance
num_layers = 1 # number of stacked LSTM layers, this is another hyperparameter that

model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss() # Mean Squared Error loss function, commonly used for regression tasks where the goal is to minimize the difference between predicted and actual values
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Adam optimizer, commonly used for training neural networks
# The Adam optimizer is an adaptive learning rate optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent, namely AdaGrad and RMSProp. It computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients. The learning rate (lr) is set to 0.01, which is a common default value for Adam, but it can be tuned based on the specific dataset and model architecture for better performance.

# 4. Train the model

num_epochs = 30 # number of times the entire training dataset will be passed through the model during training
patience = 5 # number of epochs to wait for an improvement in validation loss before stopping training, this is used for early stopping to prevent overfitting
patience_counter = 0 # counter to keep track of how many consecutive epochs have passed without improvement in validation loss, this will be reset to 0 whenever a new best model is found
best_val_loss = float('inf') # initialize the best validation loss to infinity, this will be updated during training whenever a better model is found
train_history = [] # list to store the training loss for each epoch, this can be used for plotting the training curve later to visualize how the loss changes over epochs
val_history = [] # list to store the validation loss for each epoch, this can be used for plotting the validation curve later to visualize how the loss changes over epochs and to check for overfitting or underfitting
for epoch in range(num_epochs):
    model.train() # set the model to training mode, this is important for layers like dropout and batch normalization that behave differently during training and evaluation

    epoch_loss = 0 # initialize a variable to accumulate the loss for the current epoch

    for X_batch, y_batch in train_loader: # iterate over batches of data from the DataLoader
        optimizer.zero_grad() # clear the gradients of all optimized tensors, this is necessary before calling backward() to prevent accumulation of gradients from previous iterations

        outputs = model(X_batch) # forward pass: compute predicted outputs by passing inputs to the model

        loss = criterion(outputs, y_batch.unsqueeze(1)) # compute the loss between predicted outputs and actual labels, squeeze() is used to remove any extra dimensions from the output tensor

        loss.backward() # backward pass: compute gradient of the loss with respect to model parameters

        optimizer.step() # update model parameters based on computed gradients

        epoch_loss += loss.item() * X_batch.size(0) # accumulate the total loss for the epoch, multiplying by batch size to get the total loss for all samples in the batch

    epoch_loss /= len(train_loader.dataset) # calculate average loss for the epoch by dividing total loss by number of samples in the training dataset
    train_history.append(epoch_loss) # append the average loss for the epoch to the training history list

    model.eval() # set the model to evaluation mode, this is important for layers like dropout and batch normalization that behave differently during training and evaluation
    val_loss = 0 # initialize a variable to accumulate the loss for the validation set
    with torch.no_grad(): # disable gradient calculation, this is useful for inference to save memory
        for X_val_batch, y_val_batch in val_loader:
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, y_val_batch.unsqueeze(1)).item() * X_val_batch.size(0) # accumulate the total loss for the validation set, multiplying by batch size to get the total loss for all samples in the batch
    val_loss /= len(val_loader.dataset) # calculate average loss for the validation set by dividing total loss by number of samples in the validation dataset
    val_history.append(val_loss) # append the average loss for the validation set to the validation history list
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}') # print the training and validation loss for the current epoch

    if val_loss < best_val_loss: # check if the current validation loss is better than the best validation loss seen so far
        best_val_loss = val_loss # update the best validation loss
        best_model_state = copy.deepcopy(model.state_dict()) # save the state of the model (weights and biases) that achieved the best validation loss
        patience_counter = 0 # reset the patience counter since we have a new best model
    else:
        patience_counter += 1 # increment the patience counter if the validation loss did not improve

    if patience_counter >= patience: # check if the patience counter has reached the defined patience threshold
        print("Early stopping triggered. No improvement in validation loss for 5 consecutive epochs.")
        break # stop training if the validation loss has not improved for the defined number of epochs


model.load_state_dict(best_model_state) # load the best model state (weights and biases) that was saved during training, this ensures that we use the best performing model for evaluation on the test set
print("Best model loaded for testing.")
model.eval() # set the model to evaluation mode, this is important for layers like dropout and batch normalization that behave differently during training and evaluation
with torch.no_grad(): # disable gradient calculation, this is useful for inference to save memory and
    test_predictions = model(X_test_tensor) # get predictions from the model for the test dataset
    test_loss = criterion(test_predictions, y_test_tensor.unsqueeze(1)) # compute the loss for the test dataset
    print(f'Test Loss: {test_loss.item():.4f}')
    


# ----------------------------------------
# Plotting the predicted vs actual prices for the test set
# ---------------------------------------


scaler = joblib.load('btc_scaler.pkl')

def inverse_transform( scaled_values, scaler, target_col_idx): # target_col_idx is the index of the 'log_return' price in the original Dataframe
    dummy =np.zeros((len(scaled_values),6)) # Create a dummy array with the same number of rows as the scaled values and the same number of columns as the original DataFrame
    dummy[:, target_col_idx] = scaled_values.flatten() # Assuming the target variable is at index 3, adjust if necessary
    return scaler.inverse_transform(dummy)[:,target_col_idx] # Return only the target variable after inverse transformation

actual_returns = y_test_tensor.numpy()
predicted_returns = test_predictions.numpy().flatten() # Flatten the predictions to a 1D array for easier comparison and metric calculation

last_close_normalized = X_test[:,-1, 3]
last_close_real = inverse_transform( last_close_normalized, scaler, target_col_idx=3) # target_col_idx is the index of the 'close' price in the original DataFrame, adjust if necessary

actual_prices = last_close_real * np.exp(actual_returns) # Convert log returns back to price by multiplying the last close price with the exponential of the log return
predicted_prices = last_close_real * np.exp(predicted_returns)

# ----------------------------------------
# METRICS: Calculate MAPE and RMSE for the test set
# ----------------------------------------
mape_returns = np.mean(np.abs((actual_returns - predicted_returns) / np.where(actual_returns != 0, actual_returns, 1e-8))) * 100
rmse_returns = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))
mape_prices  = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
rmse_prices  = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))

print("\n" + "="*60)
print("STANDALONE LSTM RESULTS (RETURNS-BASED MODEL)")
print("="*60)
print(f"MAPE  (returns):       {mape_returns:.4f}%")
print(f"RMSE  (returns):       {rmse_returns:.6f}")
print(f"MAPE  (prices):        {mape_prices:.4f}%")
print(f"RMSE  (prices):        ${rmse_prices:.2f}")
     

# --------------------------------------------------------
# GRPAHS TO CHECK THE MDOELS PERFORMANCE
# --------------------------------------------------------

#---------------------------------------------------------
# PLotting the training and validation loss curves
#---------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(train_history, label="Train Loss")
plt.plot(val_history, label = "Validation Loss", color='orange', linestyle='--')
plt.title("LSTM training and validation error")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show() 

# --- Predicted vs Actual Prices ---
# CHANGE 4 continued — now plotting reconstructed prices
# not raw log return values
plt.figure(figsize=(12, 6))
plt.plot(actual_prices,    label='Actual Prices',    color='blue')
plt.plot(predicted_prices, label='Predicted Prices', color='orange', linestyle='--')
plt.title('LSTM (Returns-Based): Actual vs Predicted Bitcoin Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('lstm_returns_price_prediction.png', dpi=150, bbox_inches='tight')
plt.show()
      


# Tensor must converted to numpy before inverse transformation.

plt.figure(figsize=(12, 6))
plt.plot(actual_returns, label="Actual Returns", color='blue')
plt.plot(predicted_returns, label="Predicted Returns", color='orange', linestyle='--')
plt.title("Actual vs Predicted Bitcoin Returns")
plt.xlabel("Time Steps")
plt.ylabel("Return")
plt.legend()
plt.show()

# ----------------------------------------
# Plotting the residuals (errors) to check for patterns
# ----------------------------------------

print("Plotting residuals...")
residuals = actual_returns - predicted_returns
plt.figure(figsize=(12, 6))
plt.plot(residuals, label="Residuals", color='red')
plt.axhline(0, color='black', linestyle='--') # Add a horizontal line at y=0 to visualize how residuals are distributed around zero
plt.title("Residuals Plot")
plt.xlabel("Time Steps")
plt.ylabel("Error")
plt.legend()
plt.show()

#----------------------------------------------------
# Directional Accuracy: Check if the model correctly predicts the direction of price movement
#----------------------------------------------------

def plot_directional_accuracy(actual, predicted, window=50):
    # 1. Calculate the direction of change (True if Up, False if Down)
    actual_dir = actual > 0 
    pred_dir = predicted > 0
             
    # 2. Check where they match
    hits = (actual_dir == pred_dir).astype(int) # 1 for True and 0 for False, this will give us a binary array where 1 indicates a correct directional prediction and 0 indicates an incorrect prediction
        
    # 3. Calculate a rolling average to see accuracy over time
    rolling_hit_rate = pd.Series(hits).rolling(window=window).mean()
    
    # 4. Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(rolling_hit_rate, label=f'Rolling {window}-period Accuracy', color="#1cce1c")
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random Guess (50%)')  
    
    # Add a fill to highlight areas where the model is "winning" 
    plt.fill_between(range(len(rolling_hit_rate)), rolling_hit_rate, 0.5, 
                     where=(rolling_hit_rate > 0.5), color='green', alpha=0.1)
    
    plt.title("Model Directional Accuracy (Hit Rate)")
    plt.ylabel("Accuracy %")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"Overall Mean Directional Accuracy: {np.mean(hits)*100:.2f}%")

# Usage:
plot_directional_accuracy(actual_returns, predicted_returns)  
                 
 
# -----------------------------------------
# Forecasting single future step  
# -----------------------------------------

def forecast_next_step(model, scaler, last_window_data):
    model.eval()
    
    # 1. Format input tensor shape to (1, Lookback, Features)
    input_tensor = torch.tensor(last_window_data, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        normalized_pred_value = model(input_tensor).item()
    
    # Return output is already an unscaled log return decimal
    predicted_return = normalized_pred_value
    
    # Extract and invert the last close price (Index 3 from the 6-column scaler configuration)
    last_close_norm = last_window_data[-1, 3]
    dummy = np.zeros((1, 6))
    dummy[0, 3] = last_close_norm
    real_last_close = scaler.inverse_transform(dummy)[0, 3]
    
    # Calculate target price projection
    predicted_price = real_last_close * np.exp(predicted_return)
    return predicted_price, predicted_return

# Execute forward projection step using the last test set matrix window
next_price, next_return = forecast_next_step(model, scaler, X_test[-1])
print(f"\n🚀 Predicted Next Candle Close Price Projection: ${next_price:,.2f}")
print(f"🚀 Predicted Next Period Log Return Projection: {next_return:.6f}")

#-------------------------------------------
# Save the trained model for future use
#-------------------------------------------

import json

# Save deployment files
torch.save(model.state_dict(), 'final_lstm_model.pth')
metrics = {
    "MAPE_returns": float(mape_returns), 
    "RMSE_returns": float(rmse_returns), 
    "MAPE_prices": float(mape_prices), 
    "RMSE_prices": float(rmse_prices)
}
with open('lstm_standalone_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print("\nAll pipeline execution files saved cleanly.")

print("  final_lstm_model.pth")
print("  lstm_standalone_metrics.json")
print("  lstm_actual/predicted_prices.npy")
print("  lstm_actual/predicted_returns.npy")
