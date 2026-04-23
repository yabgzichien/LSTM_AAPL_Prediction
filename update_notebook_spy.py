import json

file_path = r'c:\Users\yang\Desktop\xmum\202604\DeepLearning\StockPrediction\stock_prediction.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 4 (Index 4) - Data Acquisition
new_cell_4 = """tickers = ['AAPL', 'SPY']
data = yf.download(tickers, start='2015-01-01', end='2024-01-01')

df = pd.DataFrame()
# Reconstruct AAPL columns
df['Open'] = data['Open']['AAPL']
df['High'] = data['High']['AAPL']
df['Low'] = data['Low']['AAPL']
df['Close'] = data['Close']['AAPL']
df['Volume'] = data['Volume']['AAPL']

# Extract SPY columns
df['SPY_Close'] = data['Close']['SPY']

df.dropna(inplace=True)
df.head()"""

nb['cells'][4]['source'] = [line + '\n' if i < len(new_cell_4.split('\n'))-1 else line for i, line in enumerate(new_cell_4.split('\n'))]

# Cell 6 (Index 6) - Feature Engineering
# Let's completely recreate the source to be safe
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

# 6. Target Variable & SPY
df['Return'] = df['Close'].pct_change()
df['SPY_Return'] = df['SPY_Close'].pct_change()

df.dropna(inplace=True)
df.tail()"""

nb['cells'][6]['source'] = [line + '\n' if i < len(new_cell_6.split('\n'))-1 else line for i, line in enumerate(new_cell_6.split('\n'))]

# Cell 8 (Index 8) - Features list update
new_cell_8 = """def prepare_data(df, lookback, features, target_col):
    # Use StandardScaler for better handling of Return distributions
    scaler_f = StandardScaler()
    scaled_f = scaler_f.fit_transform(df[features].values)
    
    scaler_t = StandardScaler()
    scaled_t = scaler_t.fit_transform(df[[target_col]].values)
    
    X, y = [], []
    for i in range(lookback, len(scaled_f)):
        X.append(scaled_f[i-lookback:i, :])
        y.append(scaled_t[i, 0])
        
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler_f, scaler_t

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
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber') 
    return model

features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI', 'Price_SMA_Ratio', 'SPY_Close', 'SPY_Return', 'Return']"""

nb['cells'][8]['source'] = [line + '\n' if i < len(new_cell_8.split('\n'))-1 else line for i, line in enumerate(new_cell_8.split('\n'))]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("success")
