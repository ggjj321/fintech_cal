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
    initial_stock0 = initial_cash / (priceMat[0][0] * (1 + transFeeRate))
    initial_stock1 = initial_cash / (priceMat[0][1] * (1 + transFeeRate))
    initial_stock2 = initial_cash / (priceMat[0][2] * (1 + transFeeRate))
    initial_stock3 = initial_cash / (priceMat[0][3] * (1 + transFeeRate))
    holding = {"cash" : [1000], "stock0" : [initial_stock0], "stock1" : [initial_stock1], "stock2" : [initial_stock2], "stock3" : [initial_stock3]}
    each_move = {"cash" : [[0, -1, -1, 0]], "stock0" : [[0, -1, 0, 1000]], "stock1" : [[0, -1, 1, 1000]], "stock2" : [[0, -1, 2, 1000]], "stock3" : [[0, -1, 3, 1000]]}

    for day in range(1, dataLen):
        # case cash:
        cash_possible_choise = [holding["cash"][day-nextDay], 
                                holding["stock0"][day-nextDay] * priceMat[day][0] * (1 - transFeeRate),
                                holding["stock1"][day-nextDay] * priceMat[day][1] * (1 - transFeeRate),
                                holding["stock2"][day-nextDay] * priceMat[day][2] * (1 - transFeeRate),
                                holding["stock3"][day-nextDay] * priceMat[day][3] * (1 - transFeeRate),
        ]

        max_cash = max(cash_possible_choise)
        holding["cash"].append(max_cash)
        if cash_possible_choise.index(max_cash) - 1 == -1:
            each_move["cash"].append([day, -1, -1, 0])
        else:
            each_move["cash"].append([day, cash_possible_choise.index(max_cash) - 1, -1,  max_cash])


        #case stock0
        stock0_possible_choise = [holding["cash"][day-nextDay] / (priceMat[day][0] * (1 + transFeeRate)), 
                                holding["stock0"][day-nextDay],
                                (holding["stock1"][day-nextDay] * priceMat[day][1] * (1 - transFeeRate)) / (priceMat[day][0] * (1 + transFeeRate)),
                                (holding["stock2"][day-nextDay] * priceMat[day][2] * (1 - transFeeRate)) / (priceMat[day][0] * (1 + transFeeRate)),
                                (holding["stock3"][day-nextDay] * priceMat[day][3] * (1 - transFeeRate)) / (priceMat[day][0] * (1 + transFeeRate)),
        ]

        max_stock0 = max(stock0_possible_choise)
        holding["stock0"].append(max_stock0)
        if stock0_possible_choise.index(max_stock0) - 1 == 0:
            each_move["stock0"].append([day, 0, 0, 0])
        else:
            each_move["stock0"].append([day, stock0_possible_choise.index(max_stock0) - 1, 0, max_stock0 * priceMat[day][0]])

        #case stock1
        stock1_possible_choise = [holding["cash"][day-1] / (priceMat[day][1] * (1 + transFeeRate)), 
                                (holding["stock0"][day-1] * priceMat[day][0] * (1 - transFeeRate)) / (priceMat[day][1] * (1 + transFeeRate)),
                                holding["stock1"][day-1],
                                (holding["stock2"][day-1] * priceMat[day][2] * (1 - transFeeRate)) / (priceMat[day][1] * (1 + transFeeRate)),
                                (holding["stock3"][day-1] * priceMat[day][3] * (1 - transFeeRate)) / (priceMat[day][1] * (1 + transFeeRate)),
        ]

        max_stock1 = max(stock1_possible_choise)
        holding["stock1"].append(max_stock1)
        if stock1_possible_choise.index(max(stock1_possible_choise)) - 1 == 1:
            each_move["stock1"].append([day, 1, 1, 0])
        else:
            each_move["stock1"].append([day, stock1_possible_choise.index(max_stock1) - 1, 1, max_stock1 * priceMat[day][1]])

        #case stock2
        stock2_possible_choise = [holding["cash"][day-1] / (priceMat[day][2] * (1 + transFeeRate)), 
                                (holding["stock0"][day-1] * priceMat[day][0] * (1 - transFeeRate)) / (priceMat[day][2] * (1 + transFeeRate)),
                                (holding["stock1"][day-1] * priceMat[day][1] * (1 - transFeeRate)) / (priceMat[day][2] * (1 + transFeeRate)),
                                holding["stock2"][day-1],
                                (holding["stock3"][day-1] * priceMat[day][3] * (1 - transFeeRate)) / (priceMat[day][2] * (1 + transFeeRate)),
        ]

        max_stock2 = max(stock2_possible_choise)
        holding["stock2"].append(max_stock2)
        if stock2_possible_choise.index(max_stock2) - 1 == 2:
            each_move["stock2"].append([day, 2, 2, 0])
        else:
            each_move["stock2"].append([day, stock2_possible_choise.index(max_stock2) - 1, 2, max_stock2 * priceMat[day][2]])

        #case stock3
        stock3_possible_choise = [holding["cash"][day-1] / (priceMat[day][3] * (1 + transFeeRate)), 
                                (holding["stock0"][day-1] * priceMat[day][0] * (1 - transFeeRate)) / (priceMat[day][3] * (1 + transFeeRate)),
                                (holding["stock1"][day-1] * priceMat[day][1] * (1 - transFeeRate)) / (priceMat[day][3] * (1 + transFeeRate)),
                                (holding["stock2"][day-1] * priceMat[day][2] * (1 - transFeeRate)) / (priceMat[day][3] * (1 + transFeeRate)),
                                holding["stock3"][day-1],
        ]

        max_stock3 = max(stock3_possible_choise)
        holding["stock3"].append(max_stock3)
        if stock3_possible_choise.index(max_stock3) - 1 == 3:
            each_move["stock3"].append([day, 3, 3, 0])
        else:
            each_move["stock3"].append([day, stock3_possible_choise.index(max_stock3) - 1, 3, max_stock3 * priceMat[day][3]])

    final_all_value = [holding["cash"][-1], holding["stock0"][-1] * priceMat[-1][0] * (1 - transFeeRate), holding["stock1"][-1] * priceMat[-1][1]* (1 - transFeeRate), holding["stock2"][-1] * priceMat[-1][2] * (1 - transFeeRate), holding["stock3"][-1] * priceMat[-1][3] * (1 - transFeeRate)]
    max_index = final_all_value.index(max(final_all_value)) - 1

    for back_action_index in range(dataLen-nextDay, -1 , -1):
        if max_index == -1:
            action = each_move["cash"][back_action_index]
        else:
            action = each_move[f"stock{max_index}"][back_action_index]
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