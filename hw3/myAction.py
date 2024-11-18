import numpy as np

# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell before buy to get cash each day
    # default
    cash = 1000
    hold = 0
    # user definition
    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count   
    stockHolding = np.zeros((dataLen,stockCount))  # Mat of stock holdings
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    
    for day in range( 0, dataLen-nextDay ) :
        dayPrices = priceMat[day]  # Today price of each stock
        nextDayPrices = priceMat[ day + nextDay ]  # Next day price of each stock
        
        if day > 0:
            stockHolding[day] = stockHolding[day-1]  # The stock holding from the previous action day
        
        buyStock = -1  # which stock should buy. No action when is -1
        buyPrice = 0  # use how much cash to buy
        sellStock = []  # which stock should sell. No action when is null
        sellPrice = []  # get how much cash from sell
        bestPriceDiff = 0  # difference in today price & next day price of "buy" stock
        stockCurrentPrice = 0  # The current price of "buy" stock
        
        # Check next day price to "sell"
        for stock in range(stockCount) :
            todayPrice = dayPrices[stock]  # Today price
            nextDayPrice = nextDayPrices[stock]  # Next day price
            holding = stockHolding[day][stock]  # how much stock you are holding
            
            if holding > 0 :  # "sell" only when you have stock holding
                if nextDayPrice < todayPrice*(1+transFeeRate) :  # next day price < today price, should "sell"
                    sellStock.append(stock)
                    # "Sell"
                    sellPrice.append(holding * todayPrice)
                    cash = holding * todayPrice*(1-transFeeRate) # Sell stock to have cash
                    stockHolding[day][sellStock] = 0
        
        # Check next day price to "buy"
        if cash > 0 :  # "buy" only when you have cash
            for stock in range(stockCount) :
                todayPrice = dayPrices[stock]  # Today price
                nextDayPrice = nextDayPrices[stock]  # Next day price
                
                if nextDayPrice > todayPrice*(1+transFeeRate) :  # next day price > today price, should "buy"
                    diff = nextDayPrice - todayPrice*(1+transFeeRate)
                    if diff > bestPriceDiff :  # this stock is better
                        bestPriceDiff = diff
                        buyStock = stock
                        stockCurrentPrice = todayPrice
            # "Buy" the best stock
            if buyStock >= 0 :
                buyPrice = cash
                stockHolding[day][buyStock] = cash*(1-transFeeRate) / stockCurrentPrice # Buy stock using cash
                cash = 0
                
        # Save your action this day
        if buyStock >= 0 or len(sellStock) > 0 :
            action = []
            if len(sellStock) > 0 :
                for i in range( len(sellStock) ) :
                    action = [day, sellStock[i], -1, sellPrice[i]]
                    actionMat.append( action )
            if buyStock >= 0 :
                action = [day, -1, buyStock, buyPrice]
                actionMat.append( action )
    print(actionMat)
    return actionMat

# A DP-based approach to obtain the optimal return
def myAction01(priceMat, transFeeRate):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.

    nextDay = 1
    dataLen, stockCount = priceMat.shape  # day size & stock count
    initial_cash = 1000
    holding = {
        "cash": [initial_cash],
        **{f"stock{i}": [initial_cash / (priceMat[0][i] * (1 + transFeeRate))] for i in range(stockCount)}
    }
    each_move = {
        "cash": [[0, -1, -1, 0]],
        **{f"stock{i}": [[0, -1, i, initial_cash]] for i in range(stockCount)}
    }

    for day in range(1, dataLen):
        # case cash:
        cash_choices = [holding["cash"][day - nextDay]] + [
            holding[f"stock{i}"][day - nextDay] * priceMat[day][i] * (1 - transFeeRate)
            for i in range(stockCount)
        ]

        max_cash = max(cash_choices)
        holding["cash"].append(max_cash)
        source_index = cash_choices.index(max_cash) - 1
        each_move["cash"].append([day, source_index, -1, max_cash])

        # case stocks:
        for i in range(stockCount):
            stock_choices = [
                holding["cash"][day - nextDay] / (priceMat[day][i] * (1 + transFeeRate))
            ] + [
                holding[f"stock{j}"][day - nextDay] * priceMat[day][j] * (1 - transFeeRate) / (priceMat[day][i] * (1 + transFeeRate))
                for j in range(stockCount) if j != i
            ] + [holding[f"stock{i}"][day - nextDay]]

            max_stock = max(stock_choices)
            holding[f"stock{i}"].append(max_stock)
            source_index = stock_choices.index(max_stock) - 1
            each_move[f"stock{i}"].append([day, source_index, i, max_stock * priceMat[day][i]])

    final_all_value = [
        holding["cash"][-1]
    ] + [
        holding[f"stock{i}"][-1] * priceMat[-1][i] * (1 - transFeeRate)
        for i in range(stockCount)
    ]

    max_index = final_all_value.index(max(final_all_value)) - 1

    for back_action_index in range(dataLen - nextDay, -1, -1):
        action = each_move["cash" if max_index == -1 else f"stock{max_index}"][back_action_index]
        if action[3] != 0:
            actionMat.append(action)
        max_index = action[1]

    actionMat.reverse()

    return actionMat

# An approach that allow non-consecutive K days to hold all cash without any stocks
def myAction02(priceMat, transFeeRate, K):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    return actionMat

# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):
    actionMat = []  # An k-by-4 action matrix which holds k transaction records.
    return actionMat