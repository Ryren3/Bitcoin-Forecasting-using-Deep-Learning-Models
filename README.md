# Bitcoin-Forecasting-using-Deep-Learning-Models
Bitcoin forecasting using LSTM and LSTM+XGBoost hybrid model. Including the comparision between both the models


BTC/USDT Price Forecasting
Hybrid LSTM + XGBoost Pipeline
5-Minute Candle Prediction  ·  Binance Spot Market  ·  PyTorch + XGBoost

## 1.  Project Overview
This project implements a complete end-to-end pipeline for short-term Bitcoin price forecasting using a hybrid deep learning and gradient boosting architecture. The model predicts the log return of the next 5-minute candle given a lookback window of 65 consecutive candles, which is then converted back into a projected dollar price.

•	Stage 1 — Standalone LSTM: recurrent neural network reads sequential candle data and directly predicts the next log return.
•	Stage 2 — Hybrid LSTM + XGBoost: the trained LSTM is now repurposed as a feature extractor. Its 64-dimensional hidden state is passed to XGBoost as input features for the final prediction.

Core hypothesis
The LSTM's recurrent hidden state encodes temporal patterns that XGBoost cannot discover from raw tabular data alone. Combining both models yields sequential pattern recognition from the LSTM with non-linear decision boundary flexibility from XGBoost.

## 2.  Pipeline Architecture
### 2.1  File Structure
#	File	Purpose
1	BTC_data.py	Fetches 5-minute OHLCV candles from Binance via ccxt API
2	BTC_processing.py	Feature engineering, MinMaxScaler, train/val/test split
3	BTC_sequence.py	Sliding-window sequences of 65 candles for LSTM input
4	BTC_GridSearch.py	16-combination grid search for optimal LSTM hyperparameters
5	BTC_FinalModelling.py	Standalone LSTM: train, early stop, evaluate, save weights
6	BTC_LSTM_XGBoost.py	LSTM feature extraction → XGBoost hybrid training & evaluation

### 2.2  Data Flow
Binance API
    └─► BTC_data.py           → btc_5m_ohlcv.csv
            └─► BTC_processing.py   → btc_train/val/test_processed.csv
                                    → btc_scaler.pkl
                    └─► BTC_sequence.py  → X_train/val/test.npy  +  y_train/val/test.npy
                            ├─► BTC_GridSearch.py     → final_lstm_model.pth
                            ├─► BTC_FinalModelling.py → final_lstm_model.pth
                            │                          → lstm_standalone_metrics.json
                            └─► BTC_LSTM_XGBoost.py  → xgboost_model.json
                                                       → hybrid_metrics.json

### 2.3 Data Length
Data was extracted such that it starts from 1 Jan 2024 to the present day and time. In this case, the present date was 26 May 2026. So the total data was around 2 years and 5 months. 

## 3.  Model Design
### 3.1  Feature Engineering
•	OHLCV base features: open, high, low, close, volume
•	RSI (14-period): momentum oscillator — overbought / oversold signal
•	Log return: log(close_t / close_{t-1}) — prediction target, left unscaled
•	Cyclical time encodings: sin/cos of hour-of-day and day-of-week — encodes session patterns without ordinal assumptions

['open','high','low','close','volume','rsi'] are MinMaxScaled using the training set only. Log return and the 4 cyclical columns are added after scaling, giving 11 features per timestep.

### 3.2  Sequence Construction
•	Lookback window: 65 candles = 325 minutes ≈ 5.4 hours of market context
•	Target: log return of the candle immediately following the window
•	Split: 75% train / 15% val / 10% test — chronological, never shuffled

### 3.3  LSTM Architecture
Input        (batch, 65, 11)  — 65 timesteps × 11 features
LSTM layer   hidden_size=64, num_layers=1, batch_first=True
Dropout      0.1  (applied to LSTM output)
Dense        Linear(64 → 1)
Output       predicted log return  (scalar)

### 3.4  Hyperparameter Tuning
Grid search over 16 combinations before final training:
hidden_size  : [32, 64]
num_layers   : [1, 2]
learning_rate: [0.01, 0.001]
batch_size   : [32]
dropout      : [0.1, 0.2]
Each combination uses early stopping (patience = 5). Best weights preserved via copy.deepcopy(model.state_dict()) — avoiding the shallow-copy corruption common in naive PyTorch implementations.

### 3.5  Hybrid Architecture
LSTM (trained)  →  LSTMFeatureExtract  →  hidden state (64,)
                                               │
                                          XGBRegressor
                                          ├ n_estimators : best_iteration (early stopping)
                                          ├ max_depth    : 6
                                          ├ learning_rate: 0.03
                                          ├ subsample    : 0.8
                                          └ colsample_bytree: 0.8
Two-stage XGBoost training: Stage 1 finds optimal tree count on a clean train/val split. Stage 2 retrains with that count on combined train+val, ensuring all non-test data contributes to the final model.

## 4.  Results
### 4.1  Performance Metric Matrix
| Evaluation Metric | Standalone LSTM | Hybrid LSTM + XGBoost |
|---|---|---|
| Dataset (Train / Val / Test)	| 189k / 37.8k / 25.1k	| 189k / 37.8k / 25.1k |
| Price RMSE (Dollar Error) | $95.15 |	$95.19|
| Price MAPE |	0.0843% |	0.0843%|
| Directional Accuracy |	50.24%	| 49.67% |
| Log Returns RMSE |	0.001323 |	0.001580  (Flatline Baseline) |

### 4.2  Interpretation
Price RMSE and MAPE
Both models achieve a Price MAPE of 0.0843% and RMSE of approximately $95 — a typical dollar error of ~$95 per 5-minute candle against a BTC price of ~$76,800. The two models perform nearly identically at the price level.

Directional Accuracy
Standalone LSTM: 50.24% (just above random). Hybrid: 49.67% (just below). At 5-minute granularity BTC log returns are extremely noisy and close to a random walk — directional accuracy near 50% is the expected and honest outcome, not a model failure.

Log Returns RMSE — Flatline Baseline
The hybrid model's log returns RMSE of 0.001580 is labelled flatline because XGBoost overwhelmingly predicts values near zero — it learns the unconditional mean return is ~0 and defaults to that. The standalone LSTM's 0.001323 RMSE reflects slightly more active predictions.

Important context
5-minute BTC prediction is one of the hardest problems in quantitative finance. Returns at this frequency are noise-dominated. A directional accuracy near 50% and RMSE of ~$95 are expected outcomes. The value of this project lies in the complete, correctly implemented pipeline architecture.

## 5.  Live Forward Projection
At the final timestamp of the test matrix, the standalone LSTM generated the following forward-looking projection for the next 5-minute candle:

Projection	Value
Predicted Next Period Log Return	−0.000006
Predicted Next Candle Close Price	$76,807.25

The near-zero log return (−0.000006) is consistent with the model's mean-reversion tendency — expected behaviour when signal-to-noise ratio is low in high-frequency financial data.

## 6.  Diagnostic Plots
### 6.1  Standalone LSTM Model

Training vs Validation Loss
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/0011c4a2-c480-44e8-88cb-9e2e02184a48" />

 
Figure 1 — LSTM training and validation loss over 11 epochs. Train loss spikes at epoch 1 due to the high learning rate (0.01) then descends sharply. Early stopping triggered at epoch 11. Both curves converge near zero indicating stable training.

Actual vs Predicted Bitcoin Prices
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/5f880620-cfe9-409e-9361-6d4480b0180a" />

 
Figure 2 — LSTM price reconstruction over the full test set (~25,000 candles). Predicted prices (orange dashed) track actual prices (blue) closely across the $63,000–$83,000 range, confirming the model captures macro trend direction.

Actual vs Predicted Log Returns
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/6334cf2e-0212-4812-b8a2-f4ca7ad028f5" />

 
Figure 3 — LSTM log return predictions (orange dashed) compared to actual returns (blue). Predictions are consistently near zero — the model learns the unconditional mean return rather than individual candle movements. The large spike near timestep 6,500 is a flash-crash event.

Residuals Plot
<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/72e832fe-f679-494d-85a2-2a8dff278844" />

 
Figure 4 — LSTM residuals (actual minus predicted log returns) over time. Errors are centred around zero with no visible drift or systematic pattern, indicating no persistent bias. The large spike around timestep 6,500 corresponds to a high-volatility event.

Rolling Directional Accuracy
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/919c84e3-4493-4ff1-ad25-9b9a1f5a7ec2" />

 
Figure 5 — LSTM rolling 50-period directional accuracy. The green line oscillates above and below the 50% random-guess baseline without persistent bias in either direction. Overall mean directional accuracy: 50.24%.

### 6.2  Hybrid LSTM + XGBoost Model

Actual vs Predicted Bitcoin Prices
<img width="1784" height="881" alt="image" src="https://github.com/user-attachments/assets/5ee3e347-b755-4063-8823-ce188c17cacb" />

 
Figure 6 — Hybrid model price reconstruction. Predicted prices (orange dashed) match actual prices (blue) very closely across the full range, nearly indistinguishable from the standalone LSTM at the price level — consistent with the identical MAPE of 0.0843%.

Actual vs Predicted Log Returns
<img width="1784" height="881" alt="image" src="https://github.com/user-attachments/assets/91d88b0e-7fa0-4708-b0fb-147f4abb9790" />

 
Figure 7 — Hybrid log return predictions (orange dashed) are a nearly flat line at zero throughout the test set. XGBoost has converged on predicting the unconditional mean return — the flatline baseline behaviour noted in the metrics. Actual returns (blue) show the full range of market volatility.

Residuals Plot
<img width="1784" height="881" alt="image" src="https://github.com/user-attachments/assets/09e566d0-6130-4f67-9936-e582f3f491d7" />

 
Figure 8 — Hybrid model residuals in USD. Errors are centred around zero with no visible drift. The dominant spike (~$2,600) near timestep 6,500 corresponds to the same flash-crash event seen in LSTM residuals. The predominantly red shading indicates the model consistently underpredicts on large up-moves.

Rolling Directional Accuracy
<img width="1784" height="731" alt="image" src="https://github.com/user-attachments/assets/6a6362a1-440a-424e-a360-9baf4cc3cfb1" />

 
Figure 9 — Hybrid model rolling 50-period directional accuracy. Pattern mirrors the standalone LSTM closely — oscillating around 50% with no persistent edge above the random-guess baseline. Overall directional accuracy: 49.67%.

---

## 7. Model Comparison: Standalone LSTM vs Hybrid LSTM + XGBoost

### 7.1  Performance Metric Matrix

| Evaluation Metric | Standalone LSTM | Hybrid LSTM + XGBoost |
|---|---|---|
| Dataset (Train / Val / Test) | 189k / 37.8k / 25.1k | 189k / 37.8k / 25.1k |
| Price RMSE (Dollar Error) | $95.15 | $95.19 |
| Price MAPE | 0.0843% | 0.0843% |
| Directional Accuracy | 50.24%  | 49.67% |
| Log Returns RMSE | 0.001323  | 0.001580 |

---

### 7.2 Which Model Performed Better?

**The standalone LSTM wins on every metric**, albeit by a very small margin.

- Price RMSE: LSTM is **$0.04 cheaper per candle** in dollar error
- Price MAPE: Identical at 0.0843% — neither model has an advantage here
- Directional Accuracy: LSTM is **0.57 percentage points higher** than the hybrid
- Log Returns RMSE: LSTM is **16% lower** than the hybrid (0.001323 vs 0.001580)

The most telling difference is Log Returns RMSE. The hybrid model's RMSE of 0.001580 is
essentially the **flatline baseline** — the error you would get if you predicted zero return
for every single candle. This means XGBoost, despite receiving the LSTM's learned features,
converged on predicting the unconditional mean (≈ 0) for every timestep rather than learning
any genuine directional signal.

---

### 7.3 Why Did the Standalone LSTM Perform Better?

#### 1. XGBoost Collapsed to the Mean (Flatline Problem)

The most significant issue with the hybrid model is visible in the returns plot — the predicted
returns line is a flat orange dashed line sitting at zero throughout the entire test set. XGBoost
learned that the safest prediction is always ≈ 0, because:

- 5-minute BTC log returns are extremely small in magnitude (typically ±0.001 to ±0.005)
- The distribution of returns is centred very close to zero
- XGBoost's loss function (MSE) is minimised by predicting the mean when there is no clear signal

This is a well-known failure mode called **mean collapse** or **hedging to the mean**, where a
model sacrifices directional accuracy for lower average squared error.

#### 2. The LSTM's Final Layer Had More Context Than XGBoost's Input

The standalone LSTM's fully connected output layer sees the **raw hidden state** directly and
was trained end-to-end — the LSTM learned to produce hidden states that are *specifically useful
for the final linear prediction*. In the hybrid model, the feature extractor was frozen after
standalone training. XGBoost received hidden states that were optimised for the standalone task,
not for XGBoost's tree-splitting algorithm. This domain mismatch limits what XGBoost can extract.

#### 3. The Signal-to-Noise Ratio Is Too Low for XGBoost to Add Value

XGBoost's strength is finding non-linear interactions between features when genuine signal exists.
At 5-minute frequency, BTC returns are dominated by microstructure noise. The LSTM's 64 hidden
features encode this noisy signal — XGBoost then attempts to find structure in what is
essentially noise-on-top-of-noise. In this regime, the simpler model (LSTM linear output layer)
outperforms the more complex one (XGBoost tree ensemble).

---

### 7.4 What the Graphs Confirm

| Plot | LSTM | Hybrid |
|---|---|---|
| **Price prediction** | Orange and blue lines nearly indistinguishable | Same — both track the macro trend equally |
| **Log return prediction** | Predicted line is flat but has minor variation | Predicted line is a perfectly flat zero baseline |
| **Residuals** | Noisy around zero, no drift | Same noise pattern — errors driven by the same flash-crash events |
| **Directional accuracy** | Oscillates above and below 50%, slight upward bias | Oscillates around 50% with no consistent edge |

The price plots looking identical for both models is explained by the mathematics of log return
reconstruction. Since both models predict near-zero returns, and price is reconstructed as
`last_close * exp(predicted_return)`, a near-zero return gives `exp(≈0) ≈ 1.0`, meaning the
predicted price is approximately the last known close price shifted by a tiny amount. This is
why predicted prices track actual prices visually well even though the return predictions
themselves are essentially zero — the model is exploiting the fact that BTC prices are
persistent (today's price is a good predictor of tomorrow's price at 5-minute scale).

---

### 7.5 Key Takeaways

1. **The hybrid architecture is theoretically sound** — using LSTM features as XGBoost inputs
   is a legitimate and widely used approach in time series forecasting.

2. **The bottleneck is the data frequency, not the models** — at 5-minute granularity, returns
   are too noisy for either model to extract consistent directional signal. This is not a
   pipeline failure; it is an honest reflection of the difficulty of the problem.

3. **The flatline collapse in XGBoost is informative** — it tells us the LSTM's hidden
   representations do not contain enough structured signal for a tree-based model to split on.
   This would likely improve with longer training history, more features (order book data,
   funding rates, sentiment), or a lower-frequency target (hourly or daily returns).

4. **Both models successfully capture macro price trend** — the price reconstruction plots show
   that both models correctly follow BTC's overall trajectory from ~$63,000 to ~$83,000 across
   the test period, even though candle-level directional accuracy is near random.

5. **The standalone LSTM is the better deployment choice** — simpler, faster, and marginally
   more accurate on every metric. The hybrid adds computational cost (feature extraction +
   XGBoost training) without delivering improved performance on this dataset.

---

### 7.6 How to Improve the Hybrid Model

If revisiting this project, the following changes would likely close the performance gap:

- **Lower frequency target** — train on hourly or 4-hour returns where signal-to-noise is higher
- **Richer features** — add multiple indicator set-ups like MACD, MA, etc.
- **Fine-tune LSTM jointly with XGBoost** — rather than freezing LSTM weights, train end-to-end
- **Alternative second-stage models** — LightGBM or a small MLP may generalise better than
  XGBoost on this feature set given the low signal strength

---


## 8.  Technology Stack
Category	Library	Role
Data Ingestion	ccxt	Binance API connection, OHLCV fetching
Data Processing	pandas, numpy	DataFrame operations, feature engineering
Technical Indicators	ta	RSI calculation
Scaling	scikit-learn	MinMaxScaler — fit on train only
Deep Learning	PyTorch	LSTM definition, training loop, GPU support
Gradient Boosting	XGBoost	Hybrid second-stage regressor
Model Persistence	joblib, torch	Scaler and weight saving
Visualisation	matplotlib	Training curves, price plots, residuals

## 9.  Limitations and Honest Assessment
•	5-minute returns are near-random: at this frequency BTC price changes are dominated by microstructure noise. Near-50% directional accuracy is expected.
•	Single-step prediction only: the model predicts one candle at a time. Multi-step forecasting would compound errors significantly.
•	No transaction costs modelled: directional accuracy does not account for exchange fees, bid-ask spread, or execution latency. This is a research model, not a trading system.
•	Flatline tendency in hybrid: XGBoost defaulting to near-zero predictions indicates it is learning the unconditional mean rather than capturing genuine directional signal.
•	Flash-crash sensitivity: the large spike around timestep 6,500 in both residual plots represents an extreme event the model cannot anticipate — a known limitation of all supervised return models.

## 10.  How to Run
### 1. Install dependencies
pip install ccxt pandas numpy ta scikit-learn torch xgboost matplotlib joblib

### 2. Fetch data from Binance
python BTC_data.py

### 3. Feature engineering and preprocessing
python BTC_processing.py

### 4. Create LSTM sequences
python BTC_sequence.py

### 5. Hyperparameter grid search (saves final_lstm_model.pth)
python BTC_ModellingWithGridSearch.py

### 6. Standalone LSTM training and evaluation
python BTC_FinalModelling.py

### 7. Hybrid LSTM + XGBoost training and evaluation
python BTC_Final_Modelling_LSTM_XGBoost.py

💡  GPU acceleration
BTC_ModellingWithGridSearch.py automatically detects and uses CUDA if a GPU is available. No code changes required — device is selected at runtime via torch.device('cuda' if torch.cuda.is_available() else 'cpu').


This project is for research and educational purposes only. It does not constitute financial advice and should not be used as the sole basis for any trading or investment decision.
