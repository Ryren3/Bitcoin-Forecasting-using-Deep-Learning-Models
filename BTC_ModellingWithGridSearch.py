import json
import copy
import torch
import torch.nn as nn
import numpy as np
import itertools
import time
from torch.utils.data import TensorDataset, DataLoader

# Global device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Grid search engine locked onto device: {device.type.upper()}")


# ----------------------------------------
# 1. Define the LSTM Model
# ----------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])
        return out


# ----------------------------------------
# 2. Define the Training Function (CUDA Safe)
# ----------------------------------------
def train_model(model, train_loader, val_loader, lr, num_epochs, patience):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            # PUSH TO DEVICE (Fixes device mismatch crash)
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # --- Validation phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                # PUSH TO DEVICE
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device).unsqueeze(1)

                val_predictions = model(X_val_batch)
                loss = criterion(val_predictions, y_val_batch)
                val_loss += loss.item() * X_val_batch.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        # --- Early stopping check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Copy state dict values safely back to CPU to preserve GPU VRAM
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore uncorrupted peak optimization weights
    model.load_state_dict(best_weights)
    return best_val_loss, best_weights


# ----------------------------------------
# 3. Define the Parameter Grid
# ----------------------------------------
param_grid = {
    'hidden_size': [32, 64],
    'num_layers':  [1, 2],
    'lr':          [0.01, 0.001],
    'batch_size':  [32],
    'dropout':     [0.1, 0.2],
}


# ----------------------------------------
# 4. Run the Grid Search
# ----------------------------------------
def run_grid_search(X_train, y_train, X_val, y_val, param_grid, input_size, num_epochs=30, patience=5):
    # Tensors are initialized locally on system RAM
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val)

    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    total = len(combinations)
    print(f"Total combinations to train: {total}")
    print("="*60)

    results = []
    best_overall_weights = None
    min_observed_loss = float('inf')

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        start_time = time.time()

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader  = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)

        model = LSTMModel(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device) # Move fresh network blueprint to active CUDA cluster

        best_val_loss, runtime_weights = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=params['lr'],
            num_epochs=num_epochs,
            patience=patience
        )

        if best_val_loss < min_observed_loss:
            min_observed_loss = best_val_loss
            best_overall_weights = runtime_weights
            best_params = params

        elapsed = time.time() - start_time
        results.append({
            'params': params,
            'val_loss': best_val_loss,
            'time_seconds': round(elapsed, 1)
        })

        print(f"[{idx+1}/{total}] hidden={params['hidden_size']} layers={params['num_layers']} => val_loss={best_val_loss:.6f} ({elapsed:.1f}s)")

    results.sort(key=lambda x: x['val_loss'])
    
    print("\n" + "="*60 + "\nTOP 5 COMBINATIONS:\n" + "="*60)
    for i, r in enumerate(results[:5]):
        print(f"Rank {i+1}: val_loss = {r['val_loss']:.6f} | Params: {r['params']}")

    return results, best_params, best_overall_weights


# ----------------------------------------
# 5. Execute Optimization Run
# ----------------------------------------
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val   = np.load('X_val.npy')
y_val   = np.load('y_val.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')

input_size = X_train.shape[2]

results, best_params, optimal_weights = run_grid_search(
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    param_grid=param_grid,
    input_size=input_size,
    num_epochs=30,
    patience=5
)


# ----------------------------------------
# 6. Load Best Weights to Final Model
# ----------------------------------------
# Safe, time-series approved reconstruction: We initialize the best architecture
# and insert the exact optimized weights found during the uncorrupted tuning run.
final_model = LSTMModel(
    input_size=input_size,
    hidden_size=best_params['hidden_size'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout']
).to(device)

final_model.load_state_dict(optimal_weights)


# ----------------------------------------
# 7. Evaluate on Test Set (VRAM Protected)
# ----------------------------------------
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.MSELoss()
final_model.eval()

test_loss = 0
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        # Mini-batches are sent to the GPU sequentially to prevent memory exhaustion crashes
        X_test_batch = X_test_batch.to(device)
        y_test_batch = y_test_batch.to(device).unsqueeze(1)

        y_pred_batch = final_model(X_test_batch)
        loss = criterion(y_pred_batch, y_test_batch)
        test_loss += loss.item() * X_test_batch.size(0)

avg_test_loss = test_loss / len(test_loader.dataset)
print("\n" + "="*60)
print(f"🔥 FINAL UNBIASED TEST LOSS: {avg_test_loss:.6f}")
print("="*60)

# Export weights to local storage disk for production script use
torch.save(final_model.state_dict(), 'final_lstm_model.pth')
print("\nGrid search processing completed. Optimized model saved to disk.")