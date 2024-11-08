import sys
import numpy as np
import pandas as pd

# Decision of the current day by the current price, with 3 modifiable parameters
# def myStrategy(pastPriceVec, currentPrice, windowSize, alpha, beta):
# 	action=0		# action=1(buy), -1(sell), 0(hold), with 0 as the default action
# 	dataLen=len(pastPriceVec)		# Length of the data vector
# 	if dataLen==0:
# 		return action
# 	# Compute ma
# 	if dataLen<windowSize:
# 		ma=np.mean(pastPriceVec)	# If given price vector is small than windowSize, compute MA by taking the average
# 	else:
# 		windowedData=pastPriceVec[-windowSize:]		# Compute the normal MA using windowSize
# 		ma=np.mean(windowedData)
# 	# Determine action
# 	if (currentPrice-ma)>alpha:		# If price-ma > alpha ==> buy
# 		action=1
# 	elif (currentPrice-ma)<-beta:	# If price-ma < -beta ==> sell
# 		action=-1
# 	return action

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


def compute_kd(price_vec, period=14):
    if len(price_vec) < period:
        return None, None  # Not enough data to calculate KD
    
    low_min = np.min(price_vec[-period:])
    high_max = np.max(price_vec[-period:])
    
    if high_max - low_min == 0:
        return 50, 50  # Avoid division by zero, return neutral values
    
    k = 100 * (price_vec[-1] - low_min) / (high_max - low_min)
    d = k  # Initialize D as K for the first calculation
    return k, d

# def myStrategy(pastPriceVec, currentPrice, short_rsi_period=14, long_rsi_period=28):
#     action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
#     dataLen = len(pastPriceVec)  # Length of the data vector
#     if dataLen == 0:
#         return action

#     # Include currentPrice in RSI calculation
#     extendedPriceVec = np.append(pastPriceVec, currentPrice)
    
#     # Compute short-term and long-term RSI
#     short_rsi = compute_rsi(extendedPriceVec, short_rsi_period)
#     long_rsi = compute_rsi(extendedPriceVec, long_rsi_period)
    
#     # Determine action based on RSI crossover
#     if short_rsi is not None and long_rsi is not None:
#         if short_rsi > long_rsi and pastPriceVec[-1] < currentPrice:  # Buy if short RSI crosses above long RSI
#             action = 1
#         elif short_rsi < long_rsi and pastPriceVec[-1] > currentPrice:  # Sell if short RSI crosses below long RSI
#             action = -1

#     return action

def myStrategy(pastPriceVec, currentPrice, low_rsi_threshold=28, heigh_rsi_threshold=78, short_rsi_period=23, long_rsi_period=178, windowSize=11, alpha=0, beta=1):
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
        if (short_rsi > long_rsi and (currentPrice - ma) > alpha) or short_rsi < low_rsi_threshold:  # Buy if short RSI crosses above long RSI and price is significantly above MA
            action = 1
        elif (short_rsi < long_rsi and (currentPrice - ma) < -beta) or short_rsi > heigh_rsi_threshold:  # Sell if short RSI crosses below long RSI and price is significantly below MA
            action = -1

    return action

# Compute return rate over a given price vector, with 3 modifiable parameters
def computeReturnRate(priceVec, low_rsi_threshold=28, heigh_rsi_threshold=78, short_rsi_period=23, long_rsi_period=178, windowSize=11, alpha=0, beta=1):
	capital=1000	# Initial available capital
	capitalOrig=capital	 # original capital
	dataCount=len(priceVec)				# day size
	suggestedAction=np.zeros((dataCount,1))	# Vec of suggested actions
	stockHolding=np.zeros((dataCount,1))  	# Vec of stock holdings
	total=np.zeros((dataCount,1))	 	# Vec of total asset
	realAction=np.zeros((dataCount,1))	# Real action, which might be different from suggested action. For instance, when the suggested action is 1 (buy) but you don't have any capital, then the real action is 0 (hold, or do nothing). 
	# Run through each day
	for ic in range(dataCount):
		currentPrice=priceVec[ic]	# current price
		suggestedAction[ic]=myStrategy(priceVec[0:ic], currentPrice, windowSize=windowSize, alpha=alpha, beta=beta)		# Obtain the suggested action
		# get real action by suggested action
		if ic>0:
			stockHolding[ic]=stockHolding[ic-1]	# The stock holding from the previous day
		if suggestedAction[ic]==1:	# Suggested action is "buy"
			if stockHolding[ic]==0:		# "buy" only if you don't have stock holding
				stockHolding[ic]=capital/currentPrice # Buy stock using cash
				capital=0	# Cash
				realAction[ic]=1
		elif suggestedAction[ic]==-1:	# Suggested action is "sell"
			if stockHolding[ic]>0:		# "sell" only if you have stock holding
				capital=stockHolding[ic]*currentPrice # Sell stock to have cash
				stockHolding[ic]=0	# Stocking holding
				realAction[ic]=-1
		elif suggestedAction[ic]==0:	# No action
			realAction[ic]=0
		else:
			assert False
		total[ic]=capital+stockHolding[ic]*currentPrice	# Total asset, including stock holding and cash 
	returnRate=(total[-1].item()-capitalOrig)/capitalOrig		# Return rate of this run
	return returnRate

if __name__=='__main__':
	returnRateBest=-1.00	 # Initial best return rate
	df=pd.read_csv(sys.argv[1])	# read stock file
	adjClose=df["Adj Close"].values		# get adj close as the price vector
 
	windowSizeMin=11; windowSizeMax=25;	# Range of windowSize to explore
	alphaMin=-10; alphaMax=5;			# Range of alpha to explore
	betaMin=-5; betaMax=5				# Range of beta to explore
	# Start exhaustive search
	for windowSize in range(windowSizeMin, windowSizeMax+1):		# For-loop for windowSize
		print("windowSize=%d" %(windowSize))
		for alpha in range(alphaMin, alphaMax+1):	    	# For-loop for alpha
			print("\talpha=%d" %(alpha))
			for beta in range(betaMin, betaMax+1):		# For-loop for beta
				print("\t\tbeta=%d" %(beta), end="")	# No newline
				returnRate=computeReturnRate(adjClose, windowSize=windowSize, alpha=alpha, beta=beta)		# Start the whole run with the given parameters
				print(" ==> returnRate=%f " %(returnRate))
				if returnRate > returnRateBest:		# Keep the best parameters
					windowSizeBest=windowSize
					alphaBest=alpha
					betaBest=beta
					returnRateBest=returnRate
     
	print("Best settings: windowSize=%d, alpha=%d, beta=%d ==> returnRate=%f" %(windowSizeBest,alphaBest,betaBest,returnRateBest))		# Print the best result
	# returnRate=computeReturnRate(adjClose)
 
	# kd_period_min = 200; kd_period_max = 300
 
	# for kd_period in range(kd_period_min, kd_period_max+1):		# For-loop for windowSize
	# 	print("kd_period=%d" %(kd_period))
	# 	returnRate=computeReturnRate(adjClose, kd_period=kd_period)
	# 	print(" ==> returnRate=%f " %(returnRate))
	# 	if returnRate > returnRateBest:		# Keep the best parameters
	# 		best_kd_period = kd_period
	# 		returnRateBest=returnRate
	# print("Best settings: kd_period=%d ==> returnRate=%f" %(best_kd_period,returnRateBest))
	
	# short_rsi_peroid_min = 17; short_rsi_peroid_max = 35
	# long_rsi_peroid_min = 160; long_rsi_peroid_max = 185
	# for short_rsi_period in range(short_rsi_peroid_min, short_rsi_peroid_max+1):
	# 	print("\tshort_rsi_period=%d" %(short_rsi_period))
	# 	for long_rsi_period in range(long_rsi_peroid_min, long_rsi_peroid_max+1):
	# 		print("\tlong_rsi_period=%d" %(long_rsi_period))
	# 		returnRate=computeReturnRate(adjClose, short_rsi_period=short_rsi_period, long_rsi_period=long_rsi_period)		# Start the whole run with the given parameters
	# 		print(" ==> returnRate=%f " %(returnRate))
	# 		if returnRate > returnRateBest:		# Keep the best parameters
	# 				short_rsi_period_best = short_rsi_period
	# 				long_rsi_period_best = long_rsi_period
	# 				returnRateBest=returnRate
	
	# print("Best settings: short_rsi_period=%d, long_rsi_period=%d ==> returnRate=%f" %(short_rsi_period_best, long_rsi_period_best,returnRateBest))		# Print the best result
	
	# low_rsi_threshold_min = 5; short_rsi_threshold_max = 35
	# heigh_rsi_threshold_min = 70; heigh_rsi_threshold_max = 100
	# for low_rsi_threshold in range(low_rsi_threshold_min, short_rsi_threshold_max+1):
	# 	print("\tlow_rsi_threshold=%d" %(low_rsi_threshold))
	# 	for heigh_rsi_threshold in range(heigh_rsi_threshold_min, heigh_rsi_threshold_max+1):
	# 		print("\theigh_rsi_threshold=%d" %(heigh_rsi_threshold))
	# 		returnRate=computeReturnRate(adjClose)		# Start the whole run with the given parameters
	# 		print(" ==> returnRate=%f " %(returnRate))
	# 		if returnRate > returnRateBest:		# Keep the best parameters
	# 				low_rsi_threshold_best = low_rsi_threshold
	# 				heigh_rsi_threshold_best = heigh_rsi_threshold
	# 				returnRateBest=returnRate
	
	# print("Best settings: low_rsi_threshold_best=%d, heigh_rsi_threshold_best=%d ==> returnRate=%f" %(low_rsi_threshold_best, heigh_rsi_threshold_best,returnRateBest))		# Print the best result