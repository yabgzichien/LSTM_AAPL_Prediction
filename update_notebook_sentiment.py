import json
import textwrap

file_path = r'c:\Users\yang\Desktop\xmum\202604\DeepLearning\StockPrediction\stock_prediction.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_source(text):
    lines = text.split('\n')
    return [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]

# ============================================================
# Cell 5 (markdown) - Update section header to mention Sentiment
# ============================================================
new_cell_5 = """## 3. Advanced Feature Engineering
We are adding:
- **Bollinger Bands**: To capture volatility.
- **VROC (Volume Rate of Change)**: To capture volume momentum.
- **Price/SMA Ratios**: To normalize the trend position.
- **VWAP**: Volume Weighted Average Price for institutional price levels.
- **Synthetic Sentiment Index**: A composite 0-100 index built from VIX, momentum, RSI, and volatility — mimicking CNN's Fear & Greed Index.
- **Percentage Returns**: As the stationary target."""

nb['cells'][5]['source'] = make_source(new_cell_5)

# ============================================================
# Cell 6 (code) - Add Sentiment Index after VWAP
# ============================================================
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

# 6. Rolling VWAP (20-day window)
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
df['VWAP'] = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

# 7. VWAP Ratio (normalized distance from VWAP)
df['VWAP_Ratio'] = df['Close'] / df['VWAP']

# ============================================================
# 8. Synthetic Sentiment Index (0-100, Fear → Greed)
#    Mimics CNN Fear & Greed using 4 components:
# ============================================================

# Component 1: VIX (inverted — low VIX = greed)
vix_min = df['VIX_Close'].rolling(126).min()
vix_max = df['VIX_Close'].rolling(126).max()
sent_vix = 100 * (1 - (df['VIX_Close'] - vix_min) / (vix_max - vix_min + 1e-8))

# Component 2: Market Momentum (Close vs 125-day SMA)
sma_125 = df['Close'].rolling(125).mean()
sent_momentum = np.clip(((df['Close'] / sma_125) - 1) * 500 + 50, 0, 100)

# Component 3: RSI (already 0-100 scale, >50 = bullish)
sent_rsi = df['RSI']

# Component 4: Volatility (BB width inverted — narrow bands = greed)
bb_width = (df['Upper_BB'] - df['Lower_BB']) / df['SMA_20']
bb_min = bb_width.rolling(126).min()
bb_max = bb_width.rolling(126).max()
sent_vol = 100 * (1 - (bb_width - bb_min) / (bb_max - bb_min + 1e-8))

# Composite: equal-weighted average of all 4 components
df['Sentiment_Index'] = (sent_vix + sent_momentum + sent_rsi + sent_vol) / 4

print("Sentiment Index Statistics:")
print(df['Sentiment_Index'].describe())

# 9. Target Variable & Macro Indicators
df['Return'] = df['Close'].pct_change()
df['TNX_Return'] = df['TNX_Close'].pct_change()
df['VIX_Return'] = df['VIX_Close'].pct_change()

df.dropna(inplace=True)
print(f"\\nDataset shape after feature engineering: {df.shape}")
df.tail()"""

nb['cells'][6]['source'] = make_source(new_cell_6)

# ============================================================
# Cell 8 (code) - Update correlation heatmap to include Sentiment
# ============================================================
new_cell_8 = """# Correlation Heatmap
plt.figure(figsize=(18, 14))
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50',
             'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI',
             'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'Sentiment_Index',
             'TNX_Close', 'VIX_Close', 'Return']
corr_matrix = df[corr_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print highly correlated pairs (|r| > 0.8)
print("\\nHighly Correlated Feature Pairs (|r| > 0.8):")
print("=" * 50)
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            print(f"  {corr_matrix.columns[i]:20s} <-> {corr_matrix.columns[j]:20s}  r = {corr_matrix.iloc[i, j]:.3f}")"""

nb['cells'][8]['source'] = make_source(new_cell_8)

# ============================================================
# Cell 10 (code) - Add Sentiment_Index to features_list
# ============================================================
old_cell_10 = ''.join(nb['cells'][10]['source'])
old_features = "features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI', 'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'TNX_Close', 'TNX_Return', 'VIX_Close', 'VIX_Return', 'Return']"
new_features = "features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI', 'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'Sentiment_Index', 'TNX_Close', 'TNX_Return', 'VIX_Close', 'VIX_Return', 'Return']"

new_cell_10 = old_cell_10.replace(old_features, new_features)
nb['cells'][10]['source'] = make_source(new_cell_10)

# Verify the replacement worked
assert 'Sentiment_Index' in ''.join(nb['cells'][10]['source']), "ERROR: features_list not updated!"

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("success")
print(f"Features list now has 'Sentiment_Index': {'Sentiment_Index' in new_features}")
