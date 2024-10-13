import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_rsi(price_vec, period=14):
    if len(price_vec) < period:
        return None  # Not enough data to calculate RSI
    
    deltas = np.diff(price_vec)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100  # If no losses, RSI is 100 (strongly overbought)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load CSV data and plot Adj Close, RSI, and MA
df = pd.read_csv('price3000open.csv')

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Calculate short-term and long-term RSI
short_rsi_period = 23
long_rsi_period = 184
df['Short_RSI'] = df['Adj Close'].rolling(window=short_rsi_period).apply(lambda x: compute_rsi(x, period=short_rsi_period))
df['Long_RSI'] = df['Adj Close'].rolling(window=long_rsi_period).apply(lambda x: compute_rsi(x, period=long_rsi_period))

# Calculate Moving Average (MA)
windowSize = 120
df['MA'] = df['Adj Close'].rolling(window=windowSize).mean()

# Plotting the data
plt.figure(figsize=(14, 8))

# Plot Adjusted Close Price
plt.plot(df['Date'], df['Adj Close'], label='Adjusted Close Price', linestyle='-', marker='o')

# Plot Moving Average
plt.plot(df['Date'], df['MA'], label=f'{windowSize}-Day Moving Average', color='orange', linestyle='--')

# Plot short-term and long-term RSI on a secondary y-axis
ax1 = plt.gca()  # Get current axis
ax2 = ax1.twinx()  # Create a twin axis that shares the same x-axis
ax2.plot(df['Date'], df['Short_RSI'], label='Short-Term RSI', color='green', linestyle='-', alpha=0.6)
ax2.plot(df['Date'], df['Long_RSI'], label='Long-Term RSI', color='purple', linestyle='-', alpha=0.6)
ax2.set_ylabel('RSI')

# Set title and labels
plt.title('Stock Price with Moving Average and RSI Over Time')
ax1.set_xlabel('Date')
ax1.set_ylabel('Adjusted Close Price')
plt.xticks(rotation=45)
ax1.grid(True)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display plot
plt.tight_layout()
plt.show()
