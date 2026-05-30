# Bitcoin-Forecasting-using-Deep-Learning-Models
Bitcoin forecasting using LSTM and LSTM+XGBoost hybrid model. Including the comparision between both the models


BTC/USDT Price Forecasting
Hybrid LSTM + XGBoost Pipeline
5-Minute Candle Prediction  ·  Binance Spot Market  ·  PyTorch + XGBoost

# 1.  Project Overview
This project implements a complete end-to-end pipeline for short-term Bitcoin price forecasting using a hybrid deep learning and gradient boosting architecture. The model predicts the log return of the next 5-minute candle given a lookback window of 65 consecutive candles, which is then converted back into a projected dollar price.

•	Stage 1 — Standalone LSTM: recurrent neural network reads sequential candle data and directly predicts the next log return.
•	Stage 2 — Hybrid LSTM + XGBoost: the trained LSTM is repurposed as a feature extractor. Its 64-dimensional hidden state is passed to XGBoost as input features for the final prediction.

🎯  Core hypothesis
The LSTM's recurrent hidden state encodes temporal patterns that XGBoost cannot discover from raw tabular data alone. Combining both models yields sequential pattern recognition from the LSTM with non-linear decision boundary flexibility from XGBoost.

# 2.  Pipeline Architecture
# 2.1  File Structure
#	File	Purpose
1	BTC_data.py	Fetches 5-minute OHLCV candles from Binance via ccxt API
2	BTC_processing.py	Feature engineering, MinMaxScaler, train/val/test split
3	BTC_sequence.py	Sliding-window sequences of 65 candles for LSTM input
4	BTC_GridSearch.py	16-combination grid search for optimal LSTM hyperparameters
5	BTC_FinalModelling.py	Standalone LSTM: train, early stop, evaluate, save weights
6	BTC_LSTM_XGBoost.py	LSTM feature extraction → XGBoost hybrid training & evaluation

# 2.2  Data Flow
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

# 2.3 Data Length
Data was extracted such that it starts from 1 Jan 2024 to the present day and time. In this case, the present date was 26 May 2026. So the total data was around 2 years and 5 months. 

# 3.  Model Design
# 3.1  Feature Engineering
•	OHLCV base features: open, high, low, close, volume
•	RSI (14-period): momentum oscillator — overbought / oversold signal
•	Log return: log(close_t / close_{t-1}) — prediction target, left unscaled
•	Cyclical time encodings: sin/cos of hour-of-day and day-of-week — encodes session patterns without ordinal assumptions

['open','high','low','close','volume','rsi'] are MinMaxScaled using the training set only. Log return and the 4 cyclical columns are added after scaling, giving 11 features per timestep.

# 3.2  Sequence Construction
•	Lookback window: 65 candles = 325 minutes ≈ 5.4 hours of market context
•	Target: log return of the candle immediately following the window
•	Split: 75% train / 15% val / 10% test — chronological, never shuffled

# 3.3  LSTM Architecture
Input        (batch, 65, 11)  — 65 timesteps × 11 features
LSTM layer   hidden_size=64, num_layers=1, batch_first=True
Dropout      0.1  (applied to LSTM output)
Dense        Linear(64 → 1)
Output       predicted log return  (scalar)

# 3.4  Hyperparameter Tuning
Grid search over 16 combinations before final training:
hidden_size  : [32, 64]
num_layers   : [1, 2]
learning_rate: [0.01, 0.001]
batch_size   : [32]
dropout      : [0.1, 0.2]
Each combination uses early stopping (patience = 5). Best weights preserved via copy.deepcopy(model.state_dict()) — avoiding the shallow-copy corruption common in naive PyTorch implementations.

# 3.5  Hybrid Architecture
LSTM (trained)  →  LSTMFeatureExtract  →  hidden state (64,)
                                               │
                                          XGBRegressor
                                          ├ n_estimators : best_iteration (early stopping)
                                          ├ max_depth    : 6
                                          ├ learning_rate: 0.03
                                          ├ subsample    : 0.8
                                          └ colsample_bytree: 0.8
Two-stage XGBoost training: Stage 1 finds optimal tree count on a clean train/val split. Stage 2 retrains with that count on combined train+val, ensuring all non-test data contributes to the final model.

# 4.  Results
# 4.1  Performance Metric Matrix
Evaluation Metric	Standalone LSTM	Hybrid LSTM + XGBoost
Dataset (Train / Val / Test)	189k / 37.8k / 25.1k	189k / 37.8k / 25.1k
Price RMSE (Dollar Error)	$95.15	$95.19
Price MAPE	0.0843%	0.0843%
Directional Accuracy	50.24%	49.67%
Log Returns RMSE	0.001323	0.001580  (Flatline Baseline)

# 4.2  Interpretation
Price RMSE and MAPE
Both models achieve a Price MAPE of 0.0843% and RMSE of approximately $95 — a typical dollar error of ~$95 per 5-minute candle against a BTC price of ~$76,800. The two models perform nearly identically at the price level.

Directional Accuracy
Standalone LSTM: 50.24% (just above random). Hybrid: 49.67% (just below). At 5-minute granularity BTC log returns are extremely noisy and close to a random walk — directional accuracy near 50% is the expected and honest outcome, not a model failure.

Log Returns RMSE — Flatline Baseline
The hybrid model's log returns RMSE of 0.001580 is labelled flatline because XGBoost overwhelmingly predicts values near zero — it learns the unconditional mean return is ~0 and defaults to that. The standalone LSTM's 0.001323 RMSE reflects slightly more active predictions.

⚠️  Important context
5-minute BTC prediction is one of the hardest problems in quantitative finance. Returns at this frequency are noise-dominated. A directional accuracy near 50% and RMSE of ~$95 are expected outcomes. The value of this project lies in the complete, correctly implemented pipeline architecture.

# 5.  Live Forward Projection
At the final timestamp of the test matrix, the standalone LSTM generated the following forward-looking projection for the next 5-minute candle:

Projection	Value
Predicted Next Period Log Return	−0.000006
Predicted Next Candle Close Price	$76,807.25

The near-zero log return (−0.000006) is consistent with the model's mean-reversion tendency — expected behaviour when signal-to-noise ratio is low in high-frequency financial data.

# 6.  Diagnostic Plots
# 6.1  Standalone LSTM Model

Training vs Validation Loss
 
Figure 1 — LSTM training and validation loss over 11 epochs. Train loss spikes at epoch 1 due to the high learning rate (0.01) then descends sharply. Early stopping triggered at epoch 11. Both curves converge near zero indicating stable training.

Actual vs Predicted Bitcoin Prices
 
Figure 2 — LSTM price reconstruction over the full test set (~25,000 candles). Predicted prices (orange dashed) track actual prices (blue) closely across the $63,000–$83,000 range, confirming the model captures macro trend direction.

Actual vs Predicted Log Returns
 
Figure 3 — LSTM log return predictions (orange dashed) compared to actual returns (blue). Predictions are consistently near zero — the model learns the unconditional mean return rather than individual candle movements. The large spike near timestep 6,500 is a flash-crash event.

Residuals Plot
 
Figure 4 — LSTM residuals (actual minus predicted log returns) over time. Errors are centred around zero with no visible drift or systematic pattern, indicating no persistent bias. The large spike around timestep 6,500 corresponds to a high-volatility event.

Rolling Directional Accuracy
<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/919c84e3-4493-4ff1-ad25-9b9a1f5a7ec2" />

 
Figure 5 — LSTM rolling 50-period directional accuracy. The green line oscillates above and below the 50% random-guess baseline without persistent bias in either direction. Overall mean directional accuracy: 50.24%.

# 6.2  Hybrid LSTM + XGBoost Model

Actual vs Predicted Bitcoin Prices
 
Figure 6 — Hybrid model price reconstruction. Predicted prices (orange dashed) match actual prices (blue) very closely across the full range, nearly indistinguishable from the standalone LSTM at the price level — consistent with the identical MAPE of 0.0843%.

Actual vs Predicted Log Returns
 
Figure 7 — Hybrid log return predictions (orange dashed) are a nearly flat line at zero throughout the test set. XGBoost has converged on predicting the unconditional mean return — the flatline baseline behaviour noted in the metrics. Actual returns (blue) show the full range of market volatility.

Residuals Plot
 
Figure 8 — Hybrid model residuals in USD. Errors are centred around zero with no visible drift. The dominant spike (~$2,600) near timestep 6,500 corresponds to the same flash-crash event seen in LSTM residuals. The predominantly red shading indicates the model consistently underpredicts on large up-moves.

Rolling Directional Accuracy
 
Figure 9 — Hybrid model rolling 50-period directional accuracy. Pattern mirrors the standalone LSTM closely — oscillating around 50% with no persistent edge above the random-guess baseline. Overall directional accuracy: 49.67%.


# 7.  Technology Stack
Category	Library	Role
Data Ingestion	ccxt	Binance API connection, OHLCV fetching
Data Processing	pandas, numpy	DataFrame operations, feature engineering
Technical Indicators	ta	RSI calculation
Scaling	scikit-learn	MinMaxScaler — fit on train only
Deep Learning	PyTorch	LSTM definition, training loop, GPU support
Gradient Boosting	XGBoost	Hybrid second-stage regressor
Model Persistence	joblib, torch	Scaler and weight saving
Visualisation	matplotlib	Training curves, price plots, residuals

# 8.  Limitations and Honest Assessment
•	Short data range: ~5 months (Jan–May 2026). A robust model would train across multiple market regimes spanning several years.
•	5-minute returns are near-random: at this frequency BTC price changes are dominated by microstructure noise. Near-50% directional accuracy is expected.
•	Single-step prediction only: the model predicts one candle at a time. Multi-step forecasting would compound errors significantly.
•	No transaction costs modelled: directional accuracy does not account for exchange fees, bid-ask spread, or execution latency. This is a research model, not a trading system.
•	Flatline tendency in hybrid: XGBoost defaulting to near-zero predictions indicates it is learning the unconditional mean rather than capturing genuine directional signal.
•	Flash-crash sensitivity: the large spike around timestep 6,500 in both residual plots represents an extreme event the model cannot anticipate — a known limitation of all supervised return models.

# 9.  How to Run
# 1. Install dependencies
pip install ccxt pandas numpy ta scikit-learn torch xgboost matplotlib joblib

# 2. Fetch data from Binance
python BTC_data.py

# 3. Feature engineering and preprocessing
python BTC_processing.py

# 4. Create LSTM sequences
python BTC_sequence.py

# 5. Hyperparameter grid search (saves final_lstm_model.pth)
python BTC_ModellingWithGridSearch.py

# 6. Standalone LSTM training and evaluation
python BTC_FinalModelling.py

# 7. Hybrid LSTM + XGBoost training and evaluation
python BTC_Final_Modelling_LSTM_XGBoost.py

💡  GPU acceleration
BTC_ModellingWithGridSearch.py automatically detects and uses CUDA if a GPU is available. No code changes required — device is selected at runtime via torch.device('cuda' if torch.cuda.is_available() else 'cpu').


This project is for research and educational purposes only. It does not constitute financial advice and should not be used as the sole basis for any trading or investment decision.
