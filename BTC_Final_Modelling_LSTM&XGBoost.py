import torch
import matplotlib.pyplot as plt
import numpy as np
import joblib
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import xgboost as xgb
import json
import pandas as pd


# ----------------------------------------
# 1. Import data
# ----------------------------------------

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# ----------------------------------------
# 2. Rebuild the same (architecture) LSTM model from BTC_Final_Modelling.py
# ----------------------------------------
class LSTMModels(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModels,self).__init__() # super is used to call the parent class constructor
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self,x):
        lstm_out,_ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:,-1,:]) 
        return out
    
input_size = X_train.shape[2]
hidden_size = 64
num_layers = 1
model = LSTMModels(input_size, hidden_size, num_layers)
model.load_state_dict(torch.load('final_lstm_model.pth')) # load the best model
model.eval()

print('LSTM weights loaded successfully.')
print(f"Hidden size: {hidden_size}, Number of layers: {num_layers}, XGBoost will receive {hidden_size} features as input.")

# ----------------------------------------
# 3. Built a LSTM model that extracts features from the data and then feed those features into an XGBoost model
# ----------------------------------------

class LSTMFeatureExtract(nn.Module):
    def __init__(self, trained_model): 
        super(LSTMFeatureExtract,self).__init__() # super is used to call the parent class constructor
        self.lstm = trained_model.lstm
        self.dropout = trained_model.dropout
    def forward(self,x):
        lstm_out,_ = self.lstm(x)
        z = lstm_out[:,-1,:]
        z = self.dropout(z) # 
        return z
    
lstm_model = LSTMFeatureExtract(model)
lstm_model.eval()

def extract_features(model, X, batch_size=64):
    """
    Takes the sequence data X as numpy array, converts it to a PyTorch tensor, and passes it through the LSTM model to extract features.
    Then converts the extracted features back to a numpy array and returns it, for the XGBoost model to use.
    """
    z = []
    X_tensor = torch.from_numpy(X).float() # convert to PyTorch tensor
    with torch.no_grad(): # no need to compute gradients for feature extraction
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            z_batch = model(batch)
            z.append(z_batch.numpy())
    
    return np.vstack(z)

print("Extracting features for training set...")
XGB_train_features = extract_features(lstm_model, X_train)
print(f"Extracted features for training set: {XGB_train_features.shape}")
print("Extracting features for validation set...")
XGB_val_features = extract_features(lstm_model, X_val)
print(f"Extracted features for validation set: {XGB_val_features.shape}")
print("Extracting features for test set...")
XGB_test_features = extract_features(lstm_model, X_test)
print(f"Extracted features for test set: {XGB_test_features.shape}")

X_trainval = np.vstack((XGB_train_features, XGB_val_features))
y_trainval = np.concatenate((y_train, y_val))

print(f"Combined training and validation features for XGBoost: {X_trainval.shape}")
print(f"Combined training and validation labels for XGBoost: {y_trainval.shape}")

#---------------------------------------------------------
# 4. Train an XGBoost model on the features extracted by the LSTM
#---------------------------------------------------------

xgb_model = xgb.XGBRegressor(
    n_estimators=1000,        # Give it more room to build trees
    max_depth=10,             # Increase depth to capture non-linear feature combinations
    learning_rate=0.02,      # Lower learning rate for finer structural updates
    early_stopping_rounds=35, # Give it more patience before giving up
    subsample=0.7,           # Row subsampling to prevent noise memorization
    colsample_bytree=0.7,    # Feature subsampling
    random_state=42
)

print("Training XGBoost model...")
xgb_model.fit(
    XGB_train_features, y_train, 
    eval_set=[(XGB_val_features, y_val)], 
    verbose=50
)
best_iteration = xgb_model.get_booster().best_iteration
print(f"Best iteration from early stopping: {best_iteration}")

xgb_final = xgb.XGBRegressor(n_estimators=best_iteration, max_depth=8, learning_rate=0.02, random_state=42)

xgb_final.fit(X_trainval, y_trainval, verbose=50)

print("XGBoost training completed.")

y_pred_normalized = xgb_model.predict(XGB_test_features)
scaler   = joblib.load('btc_scaler.pkl')


def inverse_transform_column(scaled_values, scaler, col_idx):
    dummy = np.zeros((len(scaled_values), 6))
    dummy[:, col_idx] = np.array(scaled_values).flatten()
    return scaler.inverse_transform(dummy)[:, col_idx]

# Step 1 — predict and inverse transform log returns
y_pred_normalized = xgb_final.predict(XGB_test_features)


# Returns are unscaled! Extract directly without forcing through scaler
actual_returns = y_test.flatten()
predicted_returns = y_pred_normalized.flatten()

last_close_normalized = X_test[:, -1, 3]
last_close = inverse_transform_column(last_close_normalized, scaler, 3)

predicted_prices = last_close * np.exp(predicted_returns)
actual_prices = last_close * np.exp(actual_returns)

#-------------------------------------------
# 5. Evaluate the XGBoost model on the test set
#-------------------------------------------
mape_prices = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
rmse_prices = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))

print("\n" + "="*60)
print("HYBRID LSTM + XGBOOST RESULTS")
print("="*60)
print(f"MAPE Prices:          {mape_prices:.4f}%")
print(f"RMSE Prices:          ${rmse_prices:.2f}")


# ============================================================
# 11. Plots — same 4 plots as standalone for direct comparison
# ============================================================

# --- Plot 1: Predicted vs Actual ---
plt.figure(figsize=(12, 6))
plt.plot(actual_prices,    label='Actual Prices',    color='blue')
plt.plot(predicted_prices,      label='Predicted Prices', color='orange', linestyle='--')
plt.title('Hybrid LSTM+XGBoost: Actual vs Predicted Bitcoin Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('hybrid_price_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

#--- Plot 2: Predicted vs Actual Returns ---

plt.figure(figsize=(12, 6))
plt.plot(actual_returns,    label='Actual Returns',    color='blue')
plt.plot(predicted_returns,      label='Predicted Returns', color='orange', linestyle='--')
plt.title('Hybrid LSTM+XGBoost: Actual vs Predicted Bitcoin Returns')
plt.xlabel('Time Steps')
plt.ylabel('Return')
plt.legend()
plt.tight_layout()
plt.savefig('hybrid_return_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 3: Residuals ---
residuals = actual_prices - predicted_prices
plt.figure(figsize=(12, 6))
plt.plot(residuals, color='red', linewidth=0.8)
plt.axhline(0, color='black', linestyle='--')
plt.fill_between(range(len(residuals)), residuals, 0,
                 where=(residuals >= 0), color='green', alpha=0.2, label='Underpredicted')
plt.fill_between(range(len(residuals)), residuals, 0,
                 where=(residuals <  0), color='red',   alpha=0.2, label='Overpredicted')
plt.title('Hybrid LSTM+XGBoost: Residuals')
plt.xlabel('Time Steps')
plt.ylabel('Error (USD)')
plt.legend()
plt.tight_layout()
plt.savefig('hybrid_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Plot 4: Directional Accuracy ---
def plot_directional_accuracy(actual, predicted, window=50):
    actual_dir = actual > 0
    pred_dir   = predicted > 0
    hits       = (actual_dir == pred_dir).astype(int)
    rolling    = pd.Series(hits).rolling(window=window).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(rolling, label=f'Rolling {window}-period Accuracy', color='#1cce1c')
    plt.axhline(0.5, color='red', linestyle='--', label='Random Guess (50%)')
    plt.fill_between(range(len(rolling)), rolling, 0.5,
                     where=(rolling > 0.5), color='green', alpha=0.1)
    plt.title('Hybrid LSTM+XGBoost: Directional Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('hybrid_directional_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()

    overall = np.mean(hits) * 100
    print(f"Overall Directional Accuracy: {overall:.2f}%")
    return overall

directional_acc = plot_directional_accuracy(actual_returns, predicted_returns)


# ============================================================
# 12. Save everything
# ============================================================

# save metrics as JSON for comparison file
hybrid_metrics = {
    'model':             'LSTM + XGBoost Hybrid',
    'directional_acc':   round(float(directional_acc),  2),
    'best_xgb_iteration': int(xgb_model.best_iteration) if xgb_model.best_iteration is not None else "N/A"
}

with open('hybrid_metrics.json', 'w') as f:
    json.dump(hybrid_metrics, f, indent=4)

# save predictions for comparison plots
np.save('hybrid_actual_prices.npy',    actual_prices)
np.save('hybrid_predicted_prices.npy', predicted_prices)
np.save('hybrid_actual_returns.npy',   actual_returns)
np.save('hybrid_predicted_returns.npy', predicted_returns)
np.save('Z_train.npy', XGB_train_features)
np.save('Z_val.npy',   XGB_val_features)
np.save('Z_test.npy',  XGB_test_features)   # saved once only — CHANGE 8

xgb_final.save_model('xgboost_model.json')


print("\nSaved:")
print("  xgboost_model.json")
print("  hybrid_metrics.json")
print("  hybrid_y_pred_real.npy")
print("  hybrid_y_actual_real.npy")
print("  Z_train.npy / Z_val.npy / Z_test.npy")
