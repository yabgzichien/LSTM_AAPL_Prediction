import json

file_path = r'c:\Users\yang\Desktop\xmum\202604\DeepLearning\StockPrediction\stock_prediction.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_source(text):
    lines = text.split('\n')
    return [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]

markdown_content = """## 7. Trading Strategy Backtest (Financial Metrics)
We simulate a long-only trading strategy based on the model's predictions. If the model predicts 'Up' (return > 0), we buy. Otherwise, we hold cash.
We then compute:
- **Cumulative Return**: The "Equity Curve" over time.
- **Sharpe Ratio**: Risk-adjusted returns (annualized).
- **Maximum Drawdown**: The largest peak-to-trough drop in account balance."""

code_content = """# Strategy Returns (Long Only: Buy if prediction is Up)
lstm_position = (pred_lstm_ret > 0).astype(int)
gru_position = (pred_gru_ret > 0).astype(int)

# Daily returns for strategies and baseline
bnh_returns = y_test_ret
lstm_returns = lstm_position * y_test_ret
gru_returns = gru_position * y_test_ret

# Cumulative Returns
cum_bnh = np.cumprod(1 + bnh_returns)
cum_lstm = np.cumprod(1 + lstm_returns)
cum_gru = np.cumprod(1 + gru_returns)

# 1. Plot Equity Curve
plt.figure(figsize=(14, 7))
plt.plot(cum_bnh, label='Buy and Hold (Baseline)', color='black', linewidth=2)
plt.plot(cum_lstm, label='LSTM Strategy', color='blue', alpha=0.8)
plt.plot(cum_gru, label='GRU Strategy', color='green', alpha=0.8)
plt.title('Trading Strategy Backtest: Equity Curve', fontsize=16, fontweight='bold')
plt.xlabel('Test Set Timeline (Days)', fontsize=12)
plt.ylabel('Cumulative Return (1.0 = Initial Capital)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Financial Metrics calculation function
def calc_financial_metrics(returns, cum_returns, name):
    # Sharpe Ratio (annualized, assuming 252 trading days)
    # Mean daily return / daily standard deviation
    sharpe = np.sqrt(252) * (np.mean(returns) / (np.std(returns) + 1e-8))
    
    # Maximum Drawdown
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / rolling_max - 1
    mdd = np.min(drawdowns)
    
    # Total Return
    total_ret = cum_returns[-1] - 1
    
    print(f"--- {name} ---")
    print(f"Total Return : {total_ret*100:.2f}%")
    print(f"Sharpe Ratio : {sharpe:.2f}")
    print(f"Max Drawdown : {mdd*100:.2f}%\\n")

print("FINANCIAL PERFORMANCE METRICS\\n" + "="*30)
calc_financial_metrics(bnh_returns, cum_bnh, "Buy and Hold")
calc_financial_metrics(lstm_returns, cum_lstm, "LSTM Strategy")
calc_financial_metrics(gru_returns, cum_gru, "GRU Strategy")"""

# Optional: Ensure we don't append indefinitely if the user reruns
# Check if "Trading Strategy Backtest" is already in the last markdown cell
if "Trading Strategy Backtest" not in "".join(nb['cells'][-2]['source'] if len(nb['cells']) >= 2 else ""):
    nb['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": make_source(markdown_content)
    })
    
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": make_source(code_content)
    })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print("success")
else:
    print("Metrics cells already exist. No changes made.")
