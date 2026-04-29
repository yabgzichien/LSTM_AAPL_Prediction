import json

file_path = r'c:\Users\yang\Desktop\xmum\202604\DeepLearning\StockPrediction\stock_prediction.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_source(text):
    lines = text.split('\n')
    return [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]

# ============================================================
# Cell 10 (code) - Update features_list and architecture
# ============================================================
old_source = ''.join(nb['cells'][10]['source'])

# 1. Feature Pruning
# Old: ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI', 'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'Sentiment_Index', 'TNX_Close', 'TNX_Return', 'VIX_Close', 'VIX_Return', 'Return']
# We drop: 'Upper_BB', 'Lower_BB', 'RSI', 'VIX_Close', 'VIX_Return' (since Sentiment_Index captures VIX, RSI, BB width)

old_features = "features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI', 'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'Sentiment_Index', 'TNX_Close', 'TNX_Return', 'VIX_Close', 'VIX_Return', 'Return']"
new_features = "features_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'VROC', 'MACD', 'Signal_Line', 'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'Sentiment_Index', 'TNX_Close', 'TNX_Return', 'Return']"

# 2. Increase Dropout from 0.2 to 0.4
# Note: we also increase the attention dropout
new_source = old_source.replace(old_features, new_features)
new_source = new_source.replace("Dropout(0.2)", "Dropout(0.4)")

nb['cells'][10]['source'] = make_source(new_source)

# ============================================================
# Cell 8 (code) - Update correlation heatmap list to match
# ============================================================
old_cell_8 = ''.join(nb['cells'][8]['source'])
old_corr_cols = """corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50',
             'Upper_BB', 'Lower_BB', 'VROC', 'MACD', 'Signal_Line', 'RSI',
             'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 'Sentiment_Index',
             'TNX_Close', 'VIX_Close', 'Return']"""
new_corr_cols = """corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50',
             'VROC', 'MACD', 'Signal_Line', 'Price_SMA_Ratio', 'VWAP', 'VWAP_Ratio', 
             'Sentiment_Index', 'TNX_Close', 'TNX_Return', 'Return']"""
new_cell_8 = old_cell_8.replace(old_corr_cols, new_corr_cols)
nb['cells'][8]['source'] = make_source(new_cell_8)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("success")
