import numpy as np
 
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

def myStrategy(pastPriceVec, currentPrice):
    short_rsi_period=23
    long_rsi_period=178
    windowSize=11
    alpha=0
    beta=1
    
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action

    # Include currentPrice in calculations
    extendedPriceVec = np.append(pastPriceVec, currentPrice)
    
    # Compute short-term and long-term RSI
    short_rsi = compute_rsi(extendedPriceVec, short_rsi_period)
    long_rsi = compute_rsi(extendedPriceVec, long_rsi_period)
    
    # Compute Moving Average (MA)
    if dataLen < windowSize:
        ma = np.mean(extendedPriceVec)  # If given price vector is smaller than windowSize, compute MA by taking the average
    else:
        windowedData = extendedPriceVec[-windowSize:]  # Compute the normal MA using windowSize
        ma = np.mean(windowedData)
    
    # Determine action based on RSI crossover, price vs MA, and thresholds alpha/beta
    if short_rsi is not None and long_rsi is not None:
        if (short_rsi > long_rsi and (currentPrice - ma) > alpha) or short_rsi < 28:  # Buy if short RSI crosses above long RSI and price is significantly above MA
            action = 1
        elif (short_rsi < long_rsi and (currentPrice - ma) < -beta) or short_rsi > 70:  # Sell if short RSI crosses below long RSI and price is significantly below MA
            action = -1

    return action
