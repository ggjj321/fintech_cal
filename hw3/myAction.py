import numpy as np

# A simple greedy approach
def myActionSimple(priceMat, transFeeRate):
    # Explanation of my approach:
	# 1. Technical indicator used: Watch next day price
	# 2. if next day price > today price + transFee ==> buy
    #       * buy the best stock
	#    if next day price < today price + transFee ==> sell
    #       * sell if you are holding stock
    # 3. You should sell after buy to get cash each day
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

    dataLen, stockCount = priceMat.shape  # day size & stock count
    initial_cash = 1000

    holding = {'cash': [initial_cash]}
    each_move = {'cash': [[0, -1, -1, 0]]}

    # Initialize holdings and moves for each stock
    for i in range(stockCount):
        initial_stock = initial_cash / priceMat[0][i]
        holding[f'stock{i}'] = [initial_stock]
        each_move[f'stock{i}'] = [[0, -1, i, initial_cash]]

    for day in range(1, dataLen):
        # Update cash holdings
        cash_possible_choices = [holding['cash'][day - 1]]
        for i in range(stockCount):
            stock_value = holding[f'stock{i}'][day - 1] * priceMat[day][i] * (1 - transFeeRate)
            cash_possible_choices.append(stock_value)

        max_cash = max(cash_possible_choices)
        holding['cash'].append(max_cash)
        from_index = cash_possible_choices.index(max_cash) - 1  # -1 for cash, 0..stockCount-1 for stocks

        if from_index == -1:
            each_move['cash'].append([day, -1, -1, 0])
        else:
            each_move['cash'].append([day, from_index, -1, max_cash])

        # Update holdings for each stock
        for i in range(stockCount):
            stock_possible_choices = []
            # From cash to stock i
            from_cash = holding['cash'][day - 1] / (priceMat[day][i] * (1 + transFeeRate))
            stock_possible_choices.append(from_cash)
            # From stocks to stock i (including staying in the same stock)
            for j in range(stockCount):
                if j == i:
                    # Stay in the same stock
                    stock_possible_choices.append(holding[f'stock{i}'][day - 1])
                else:
                    # Sell stock j and buy stock i
                    stock_j_value = holding[f'stock{j}'][day - 1] * priceMat[day][j] * (1 - transFeeRate)
                    to_stock_i = stock_j_value / (priceMat[day][i] * (1 + transFeeRate))
                    stock_possible_choices.append(to_stock_i)

            max_stock = max(stock_possible_choices)
            holding[f'stock{i}'].append(max_stock)
            from_index = stock_possible_choices.index(max_stock)

            if from_index == i + 1:
                # Stayed in the same stock
                each_move[f'stock{i}'].append([day, i, i, 0])
            else:
                if from_index == 0:
                    from_stock = -1  # From cash
                else:
                    from_stock = from_index - 1
                action_amount = max_stock * priceMat[day][i]
                each_move[f'stock{i}'].append([day, from_stock, i, action_amount])

    # Backtracking to build actionMat
    final_values = [holding['cash'][-1]]
    for i in range(stockCount):
        stock_value = holding[f'stock{i}'][-1] * priceMat[-1][i] * (1 - transFeeRate)
        final_values.append(stock_value)

    max_final_value = max(final_values)
    max_index = final_values.index(max_final_value) - 1  # -1 for cash, 0..stockCount-1 for stocks

    for day in range(dataLen - 1, -1, -1):
        if max_index == -1:
            action = each_move['cash'][day]
        else:
            action = each_move[f'stock{max_index}'][day]
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
    class ConsecutiveData:
        def __init__(self) -> None:
            self.holding = {'cash': []}
            self.each_move = {'cash': []}

            for stock_index in range(stockCount):
                self.holding[f"stock{stock_index}"] = []
                self.each_move[f"stock{stock_index}"] = []

            self.final_income = 0
        
        def sperate_cash(self, cash, spreate_day):
            self.holding['cash'].append(cash)
            self.each_move['cash'].append([spreate_day, -1, -1, 0])

            for i in range(stockCount):
                initial_stock = cash / (priceMat[spreate_day][i] * (1 + transFeeRate))
                self.holding[f'stock{i}'].append(initial_stock)
                self.each_move[f'stock{i}'].append([spreate_day, -1, i, cash])
        
        def take_back_cash(self, back_day):
            final_cash_choise = [self.holding['cash'][-1]]

            for stock_index in range(stockCount):
                equal_value = self.holding[f'stock{stock_index}'][-1] * priceMat[back_day][stock_index] * (1 - transFeeRate)
                final_cash_choise.append(equal_value)

            max_cash = max(final_cash_choise)
            from_index = final_cash_choise.index(max_cash)

            self.holding["cash"].append(max_cash)

            if from_index == -1:
                self.each_move["cash"].append([back_day, -1, -1, 0])
            else:
                self.each_move["cash"].append([back_day, from_index, -1, max_cash])

    actionMat = []  # An k-by-4 action matrix which holds k transaction records.

    dataLen, stockCount = priceMat.shape
    initial_cash = 100

    each_consecutive_hodling_and_move = {}
    first_stage_end_time = 0
    second_stage_start_time = K 

    while second_stage_start_time < dataLen:
        each_consecutive_hodling_and_move[f"{first_stage_end_time}_{second_stage_start_time}"] = ConsecutiveData()
        consecutive_data = each_consecutive_hodling_and_move[f"{first_stage_end_time}_{second_stage_start_time}"]

        # first stage
        if first_stage_end_time > 0:
            consecutive_data.sperate_cash(initial_cash, 0)

            #dp
            for day in range(1, first_stage_end_time):
                # Update cash holdings
                cash_possible_choices = [consecutive_data.holding['cash'][day - 1]]
                for i in range(stockCount):
                    stock_value = consecutive_data.holding[f'stock{i}'][day - 1] * priceMat[day][i] * (1 - transFeeRate)
                    cash_possible_choices.append(stock_value)

                max_cash = max(cash_possible_choices)
                consecutive_data.holding['cash'].append(max_cash)
                from_index = cash_possible_choices.index(max_cash) - 1  # -1 for cash, 0..stockCount-1 for stocks

                if from_index == -1:
                    consecutive_data.each_move['cash'].append([day, -1, -1, 0])
                else:
                    consecutive_data.each_move['cash'].append([day, from_index, -1, max_cash])

                # Update holdings for each stock
                for i in range(stockCount):
                    stock_possible_choices = []
                    # From cash to stock i
                    from_cash = consecutive_data.holding['cash'][day - 1] / (priceMat[day][i] * (1 + transFeeRate))
                    stock_possible_choices.append(from_cash)
                    # From stocks to stock i (including staying in the same stock)
                    for j in range(stockCount):
                        if j == i:
                            # Stay in the same stock
                            stock_possible_choices.append(consecutive_data.holding[f'stock{i}'][day - 1])
                        else:
                            # Sell stock j and buy stock i
                            stock_j_value = consecutive_data.holding[f'stock{j}'][day - 1] * priceMat[day][j] * (1 - transFeeRate)
                            to_stock_i = stock_j_value / (priceMat[day][i] * (1 + transFeeRate))
                            stock_possible_choices.append(to_stock_i)

                    max_stock = max(stock_possible_choices)
                    consecutive_data.holding[f'stock{i}'].append(max_stock)
                    from_index = stock_possible_choices.index(max_stock)

                    if from_index == i + 1:
                        # Stayed in the same stock
                        consecutive_data.each_move[f'stock{i}'].append([day, i, i, 0])
                    else:
                        if from_index == 0:
                            from_stock = -1  # From cash
                        else:
                            from_stock = from_index - 1
                        action_amount = max_stock * priceMat[day][i]
                        consecutive_data.each_move[f'stock{i}'].append([day, from_stock, i, action_amount])


            # take back
            consecutive_data.take_back_cash(first_stage_end_time)

        # second_stage
        if first_stage_end_time != 0:
            initial_cash  = consecutive_data.holding['cash'][-1]

        consecutive_data.sperate_cash(initial_cash, second_stage_start_time)
        # dp
        for day in range(second_stage_start_time + 1, dataLen - 1):
            # Update cash holdings
            cash_possible_choices = [consecutive_data.holding['cash'][day - K - 1]]
            for i in range(stockCount):
                stock_value = consecutive_data.holding[f'stock{i}'][day - K - 1] * priceMat[day][i] * (1 - transFeeRate)
                cash_possible_choices.append(stock_value)

            max_cash = max(cash_possible_choices)
            consecutive_data.holding['cash'].append(max_cash)
            from_index = cash_possible_choices.index(max_cash) - 1  # -1 for cash, 0..stockCount-1 for stocks

            if from_index == -1:
                consecutive_data.each_move['cash'].append([day, -1, -1, 0])
            else:
                consecutive_data.each_move['cash'].append([day, from_index, -1, max_cash])

            # Update holdings for each stock
            for i in range(stockCount):
                stock_possible_choices = []
                # From cash to stock i
                from_cash = consecutive_data.holding['cash'][day - K - 1] / (priceMat[day][i] * (1 + transFeeRate))
                stock_possible_choices.append(from_cash)
                # From stocks to stock i (including staying in the same stock)
                for j in range(stockCount):
                    if j == i:
                        # Stay in the same stock
                        stock_possible_choices.append(consecutive_data.holding[f'stock{i}'][day - K - 1])
                    else:
                        # Sell stock j and buy stock i
                        stock_j_value = consecutive_data.holding[f'stock{j}'][day - K- 1] * priceMat[day][j] * (1 - transFeeRate)
                        to_stock_i = stock_j_value / (priceMat[day][i] * (1 + transFeeRate))
                        stock_possible_choices.append(to_stock_i)

                max_stock = max(stock_possible_choices)
                consecutive_data.holding[f'stock{i}'].append(max_stock)
                from_index = stock_possible_choices.index(max_stock)

                if from_index == i + 1:
                    # Stayed in the same stock
                    consecutive_data.each_move[f'stock{i}'].append([day, i, i, 0])
                else:
                    if from_index == 0:
                        from_stock = -1  # From cash
                    else:
                        from_stock = from_index - 1
                    action_amount = max_stock * priceMat[day][i]
                    consecutive_data.each_move[f'stock{i}'].append([day, from_stock, i, action_amount])
        consecutive_data.take_back_cash(dataLen - 1)
        consecutive_data.final_income = consecutive_data.holding["cash"][-1]

        first_stage_end_time += 1
        second_stage_start_time += 1

    # todo : 根據 final_income 建構 actionMat
    for consecutive_data in each_consecutive_hodling_and_move:
        print(each_consecutive_hodling_and_move[consecutive_data].holding)
        print(each_consecutive_hodling_and_move[consecutive_data].each_move)
        print(each_consecutive_hodling_and_move[consecutive_data].final_income)


    # for start_cash_holding_day in range(dataLen - K + 1):
    #     each_consecutive_hodling_and_move[f"{start_cash_holding_day}_{start_cash_holding_day + K - 1}"] = ConsecutiveData()
    #     consecutive_data = each_consecutive_hodling_and_move[f"{start_cash_holding_day}_{start_cash_holding_day + K - 1}"]

    #     consecutive_data.sperate_cash(initial_cash, start_cash_holding_day)
        

    #     # before holding
    #     for before_holding_day in range(1, start_cash_holding_day):
    #         cash_possible_choices = [consecutiveData.holding['cash'][before_holding_day - 1]]

    #         # Update cash holdings
    #         for i in range(stockCount):
    #             stock_value = consecutiveData.holding[f'stock{i}'][before_holding_day - 1] * priceMat[before_holding_day][i] * (1 - transFeeRate)
    #             cash_possible_choices.append(stock_value)

    #         max_cash = max(cash_possible_choices)
    #         consecutiveData.holding['cash'].append(max_cash)
    #         from_index = cash_possible_choices.index(max_cash) - 1  # -1 for cash, 0..stockCount-1 for stocks

    #         if from_index == -1:
    #             consecutiveData.each_move['cash'].append([before_holding_day, -1, -1, 0])
    #         else:
    #             consecutiveData.each_move['cash'].append([before_holding_day, from_index, -1, max_cash])

    #         # Update holdings for each stock
    #         for i in range(stockCount):
    #             stock_possible_choices = []

    #             # From cash to stock i
    #             from_cash = consecutiveData.holding['cash'][before_holding_day - 1] / (priceMat[before_holding_day][i] * (1 + transFeeRate))
    #             stock_possible_choices.append(from_cash)
    #             # From stocks to stock i (including staying in the same stock)
    #             for j in range(stockCount):
    #                 if j == i:
    #                     # Stay in the same stock
    #                     stock_possible_choices.append(consecutiveData.holding[f'stock{i}'][before_holding_day - 1])
    #                 else:
    #                     # Sell stock j and buy stock i
    #                     stock_j_value = consecutiveData.holding[f'stock{j}'][before_holding_day - 1] * priceMat[before_holding_day][j] * (1 - transFeeRate)
    #                     to_stock_i = stock_j_value / (priceMat[before_holding_day][i] * (1 + transFeeRate))
    #                     stock_possible_choices.append(to_stock_i)

    #             max_stock = max(stock_possible_choices)
    #             consecutiveData.holding[f'stock{i}'].append(max_stock)
    #             from_index = stock_possible_choices.index(max_stock)

    #             if from_index == i + 1:
    #                 # Stayed in the same stock
    #                 consecutiveData.each_move[f'stock{i}'].append([before_holding_day, i, i, 0])
    #             else:
    #                 if from_index == 0:
    #                     from_stock = -1  # From cash
    #                 else:
    #                     from_stock = from_index - 1
    #                 action_amount = max_stock * priceMat[before_holding_day][i]
    #                 consecutiveData.each_move[f'stock{i}'].append([before_holding_day, from_stock, i, action_amount])
            
    #     # print("cash : ")
    #     # print(consecutiveData.holding['cash'])
    #     # print(consecutiveData.each_move['cash'])

    #     # for i in range(4):
    #     #     print(f"stock{i} : ")
    #     #     print(consecutiveData.holding[f'stock{i}'])
    #     #     print(consecutiveData.each_move[f'stock{i}'])

    #     # after holding

    #     # 全部換成現金
    #     all_possible_return_cash = [consecutiveData.holding['cash'][-1]]
    #     for stock_index in range(4):
    #         all_possible_return_cash.append(consecutiveData.holding[f'stock{stock_index}'][-1] * priceMat[start_cash_holding_day][stock_index] * (1 - transFeeRate))
            
    #     max_change_to_cash = max(all_possible_return_cash)
    #     from_stock = all_possible_return_cash.index(max_change_to_cash) - 1

    #     consecutiveData.take_back_move = [start_cash_holding_day, from_stock, -1, max_change_to_cash]

    #     after_holding_day = start_cash_holding_day + K

    #     consecutiveData.holding['cash'].append(max_change_to_cash)
    #     consecutiveData.each_move['cash'].append([after_holding_day, from_stock, -1, max_change_to_cash])

    #     for stock_index in range(stockCount):
    #         consecutiveData.holding[f'stock{stock_index}'].append(max_change_to_cash / (priceMat[after_holding_day][stock_index] * (1 + transFeeRate)))
    #         consecutiveData.each_move[f'stock{stock_index}'].append([after_holding_day, -1, stock_index, max_change_to_cash])
        

    #     for after_holding_day in range(start_cash_holding_day + K + 1, dataLen):
    #         print("after" + str(after_holding_day))
    #         cash_possible_choices = [consecutiveData.holding['cash'][after_holding_day - K - 1]]

    #         # Update cash holdings
    #         for i in range(stockCount):
    #             stock_value = consecutiveData.holding[f'stock{i}'][after_holding_day - K - 1] * priceMat[after_holding_day][i] * (1 - transFeeRate)
    #             cash_possible_choices.append(stock_value)

    #         max_cash = max(cash_possible_choices)
    #         consecutiveData.holding['cash'].append(max_cash)
    #         from_index = cash_possible_choices.index(max_cash) - 1  # -1 for cash, 0..stockCount-1 for stocks

    #         if from_index == -1:
    #             consecutiveData.each_move['cash'].append([after_holding_day, -1, -1, 0])
    #         else:
    #             consecutiveData.each_move['cash'].append([after_holding_day, from_index, -1, max_cash])

    #         # Update holdings for each stock
    #         for i in range(stockCount):
    #             stock_possible_choices = []

    #             # From cash to stock i
    #             from_cash = consecutiveData.holding['cash'][after_holding_day - K - 1] / (priceMat[after_holding_day][i] * (1 + transFeeRate))
    #             stock_possible_choices.append(from_cash)
    #             # From stocks to stock i (including staying in the same stock)
    #             for j in range(stockCount):
    #                 if j == i:
    #                     # Stay in the same stock
    #                     stock_possible_choices.append(consecutiveData.holding[f'stock{i}'][after_holding_day - K - 1])
    #                 else:
    #                     # Sell stock j and buy stock i
    #                     stock_j_value = consecutiveData.holding[f'stock{j}'][after_holding_day - K - 1] * priceMat[after_holding_day][j] * (1 - transFeeRate)
    #                     to_stock_i = stock_j_value / (priceMat[after_holding_day][i] * (1 + transFeeRate))
    #                     stock_possible_choices.append(to_stock_i)

    #             max_stock = max(stock_possible_choices)
    #             consecutiveData.holding[f'stock{i}'].append(max_stock)
    #             from_index = stock_possible_choices.index(max_stock)

    #             if from_index == i + 1:
    #                 # Stayed in the same stock
    #                 consecutiveData.each_move[f'stock{i}'].append([after_holding_day, i, i, 0])
    #             else:
    #                 if from_index == 0:
    #                     from_stock = -1  # From cash
    #                 else:
    #                     from_stock = from_index - 1
    #                 action_amount = max_stock * priceMat[after_holding_day][i]
    #                 consecutiveData.each_move[f'stock{i}'].append([after_holding_day, from_stock, i, action_amount])
    #     # Backtracking to build actionMat
    #     final_values = [consecutiveData.holding['cash'][-1]]
    #     for stock_index in range(stockCount):
    #         final_values.append(consecutiveData.holding[f"stock{stock_index}"][-1] * priceMat[-1][stock_index])
        
    #     max_final_value = max(final_values)
    #     max_index = final_values.index(max_final_value) - 1  # -1 for cash, 0..stockCount-1 for stocks

    #     consecutiveData.final_income = max_final_value
    #     consecutiveData.final_index = max_index
    
    # for consecutive_data in each_consecutive_hodling_and_move:
    #     print(each_consecutive_hodling_and_move[consecutive_data].holding)
    #     print(each_consecutive_hodling_and_move[consecutive_data].each_move)
    #     print(each_consecutive_hodling_and_move[consecutive_data].final_income)
    #     print(each_consecutive_hodling_and_move[consecutive_data].final_index)
    
    # max_income_consecutive = max(each_consecutive_hodling_and_move.values(), key=lambda x: x.final_income)
    # print(max_income_consecutive.holding)
    # print(max_income_consecutive.each_move)
    # print(max_income_consecutive.final_income)
    # print(max_income_consecutive.final_index)
    # print(max_income_consecutive.take_back_move)

    return actionMat

# priceMat = [
#     [10, 20, 30, 40],
#     [11, 19, 31, 44],
#     [12, 18, 35, 47],
#     [9, 22, 29, 48],
#     [10, 20, 30, 40],
# ]
# priceMat = np.array(priceMat)
# transFeeRate = 0.01
# K = 2

# myAction03(priceMat, transFeeRate, K)