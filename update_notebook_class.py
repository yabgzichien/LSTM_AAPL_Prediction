import json

file_path = r'c:\Users\yang\Desktop\xmum\202604\DeepLearning\StockPrediction\stock_prediction.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 6 (Index 6) - Add Target_Direction
new_cell_6 = """# 1. Moving Averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# 2. Bollinger Bands
std_20 = df['Close'].rolling(window=20).std()
df['Upper_BB'] = df['SMA_20'] + (std_20 * 2)
df['Lower_BB'] = df['SMA_20'] - (std_20 * 2)

# 3. Volume Rate of Change (VROC)
df['VROC'] = df['Volume'].pct_change(periods=10)

# 4. MACD & RSI
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 5. Price-to-SMA Ratio
df['Price_SMA_Ratio'] = df['Close'] / df['SMA_20']

# 6. Target Variable & Macro Indicators
df['Return'] = df['Close'].pct_change()
df['TNX_Return'] = df['TNX_Close'].pct_change()
df['VIX_Return'] = df['VIX_Close'].pct_change()

# NEW: Classification Target (1 if Return > 0 else 0)
df['Target_Direction'] = (df['Return'] > 0).astype(int)

df.dropna(inplace=True)
df.tail()"""

nb['cells'][6]['source'] = [line + '\n' if i < len(new_cell_6.split('\n'))-1 else line for i, line in enumerate(new_cell_6.split('\n'))]

# Cell 8 (Index 8) - Modifying prepare_data, build_model, features_list
new_cell_8 = """def prepare_data(df, lookback, features, target_col):
    # Use StandardScaler for input features
    scaler_f = StandardScaler()
    scaled_f = scaler_f.fit_transform(df[features].values)
    
    # Target is already binary (0 or 1), no scaling needed
    scaled_t = df[[target_col]].values
    
    X, y = [], []
    for i in range(lookback, len(scaled_f)):
        X.append(scaled_f[i-lookback:i, :])
        y.append(scaled_t[i, 0])
        
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler_f, None

def build_model(input_shape, model_type='LSTM'):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
    else:
        model.add(Bidirectional(GRU(64, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(32))
    
    model.add(Dropout(0.2))
    # NEW: Sigmoid activation for binary probability
    model.add(Dense(1, activation='sigmoid'))
    # NEW: binary_crossentropy loss optimized for classification
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy']) 
    return model

# Removed 'Return' from features array since we use Target_Direction now as pure target
features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI', 'Price_SMA_Ratio', 'TNX_Close', 'TNX_Return', 'VIX_Close', 'VIX_Return']"""

nb['cells'][8]['source'] = [line + '\n' if i < len(new_cell_8.split('\n'))-1 else line for i, line in enumerate(new_cell_8.split('\n'))]

# Cell 10 (Index 10) - Update target_col in Tuning
old_cell_10 = "".join(nb['cells'][10]['source'])
new_cell_10 = old_cell_10.replace("'Return'", "'Target_Direction'")
nb['cells'][10]['source'] = [line + '\n' if i < len(new_cell_10.split('\n'))-1 else line for i, line in enumerate(new_cell_10.split('\n'))]

# Cell 12 (Index 12) - Update target_col in Final Training
old_cell_12 = "".join(nb['cells'][12]['source'])
new_cell_12 = old_cell_12.replace("'Return'", "'Target_Direction'")
nb['cells'][12]['source'] = [line + '\n' if i < len(new_cell_12.split('\n'))-1 else line for i, line in enumerate(new_cell_12.split('\n'))]

# Cell 14 (Index 14) - Evaluation cell rewrite
new_cell_14 = """# Predictions (Probabilities)
pred_lstm_prob = model_lstm.predict(X_test).flatten()
pred_gru_prob = model_gru.predict(X_test).flatten()

# Convert probabilities to binary outcomes (Probability > 0.5 means UP)
pred_lstm_dir = (pred_lstm_prob > 0.5).astype(int)
pred_gru_dir = (pred_gru_prob > 0.5).astype(int)
y_test_dir = y_test.flatten()

# Visuals: Prediction Probabilities Distribution
plt.figure(figsize=(10, 5))
sns.histplot(pred_lstm_prob, color='blue', alpha=0.5, label='LSTM Probabilities', bins=30)
sns.histplot(pred_gru_prob, color='green', alpha=0.5, label='GRU Probabilities', bins=30)
plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.title(f'Classification Probabilities (Lookback: {best_lookback} days)')
plt.xlabel('Probability of "UP" Move')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Metrics
def get_metrics_class(model_name, actual_dir, pred_dir):
    acc = accuracy_score(actual_dir, pred_dir)
    prec = precision_score(actual_dir, pred_dir, zero_division=0)
    rec = recall_score(actual_dir, pred_dir, zero_division=0)
    f1 = f1_score(actual_dir, pred_dir, zero_division=0)
    
    print(f"[{model_name}] Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

get_metrics_class("LSTM (Classification)", y_test_dir, pred_lstm_dir)
get_metrics_class("GRU (Classification)", y_test_dir, pred_gru_dir)
"""

nb['cells'][14]['source'] = [line + '\n' if i < len(new_cell_14.split('\n'))-1 else line for i, line in enumerate(new_cell_14.split('\n'))]


with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("success")
