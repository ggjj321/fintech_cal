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
    prices = np.array(priceMat)
    s, a = prices.shape  
    initial_cash = 1000 

    dp = np.full((s + 1, K + 1, a + 1), -np.inf, dtype=object)
    for day in range(s + 1):
        for cash_days in range(K + 1):
            for holding in range(a + 1):
                dp[day][cash_days][holding] = (-np.inf, None, None)

    dp[0][0][a] = (initial_cash, None, None)  

    for day in range(1, s + 1):
        for cash_days in range(K + 1):
            for holding in range(a + 1):
                current_value, prev_state, action = dp[day - 1][cash_days][holding]
                if current_value == -np.inf:
                    continue

                if holding == a:
                    if cash_days + 1 <= K:
                        new_cash_days = cash_days + 1
                        new_value = current_value
                        if dp[day][new_cash_days][a][0] < new_value:
                            dp[day][new_cash_days][a] = (new_value, (cash_days, holding), None)
                    else:
                        continue

                    for j in range(a):
                        stock_price = prices[day - 1][j]
                        if stock_price <= 0:
                            continue
                        new_value = (current_value * (1 - transFeeRate)) / stock_price
                        if dp[day][cash_days][j][0] < new_value:
                            dp[day][cash_days][j] = (new_value, (cash_days, holding), (day - 1, -1, j, current_value * (1 - transFeeRate)))
                else:
                    stock_price = prices[day - 1][holding]
                    new_value = current_value
                    if dp[day][cash_days][holding][0] < new_value:
                        dp[day][cash_days][holding] = (new_value, (cash_days, holding), None)

                    if cash_days + 1 <= K:
                        new_cash_days = cash_days + 1
                        sell_cash = current_value * stock_price * (1 - transFeeRate)
                        if dp[day][new_cash_days][a][0] < sell_cash:
                            dp[day][new_cash_days][a] = (sell_cash, (cash_days, holding), (day - 1, holding, -1, sell_cash))
                    else:
                        continue

                    for j in range(a):
                        if j == holding:
                            continue
                        new_stock_price = prices[day - 1][j]
                        if new_stock_price <= 0:
                            continue
                        switch_cash = current_value * stock_price * (1 - transFeeRate)
                        new_value = (switch_cash * (1 - transFeeRate)) / new_stock_price
                        if dp[day][cash_days][j][0] < new_value:
                            dp[day][cash_days][j] = (new_value, (cash_days, holding), (day - 1, holding, j, switch_cash))
    max_value = -np.inf
    end_state = None
    for cash_days in range(K, K + 1):
        for holding in range(a + 1):
            current_value, _, _ = dp[s][cash_days][holding]
            if current_value > max_value:
                max_value = current_value
                end_state = (s, cash_days, holding)

    if end_state is None:
        return [], 0  

    day, cash_days, holding = end_state
    if holding != a:
        if cash_days + 1 <= K:
            stock_price = prices[day - 1][holding]
            sell_cash = dp[day][cash_days][holding][0] * stock_price * (1 - transFeeRate)
            dp[day][cash_days + 1][a] = (sell_cash, (cash_days, holding), (day - 1, holding, -1, sell_cash))
            cash_days += 1
            holding = a
            max_value = sell_cash
            end_state = (day, cash_days, holding)
        else:
            return [], 0

    actions = []
    state = end_state
    while state[0] > 0:
        day, cash_days, holding = state
        current_value, prev_state, action = dp[day][cash_days][holding]
        if action is not None:
            actions.append(action)
        if prev_state is None:
            break
        state = (day - 1, prev_state[0], prev_state[1])

    actions.reverse()
    return actions

# An approach that allow consecutive K days to hold all cash without any stocks    
def myAction03(priceMat, transFeeRate, K):

    dataLen, stockCount = priceMat.shape
    initial_cash = 1000

    class ConsecutiveData:
        def __init__(self):
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
            from_index = final_cash_choise.index(max_cash) - 1
            self.holding["cash"].append(max_cash)
            if from_index == -1:
                self.each_move["cash"].append([back_day, -1, -1, 0])
            else:
                self.each_move["cash"].append([back_day, from_index, -1, max_cash])

    actionMat = []

    max_final_income = 0
    max_final_income_obj = None

    first_stage_end_time = 0
    second_stage_start_time = K

    while second_stage_start_time < dataLen:
        consecutive_data = ConsecutiveData()

        # stage1 
        if first_stage_end_time > 0:
            consecutive_data.sperate_cash(initial_cash, 0)
            for day in range(1, first_stage_end_time):
                # update cash
                cash_possible_choices = [consecutive_data.holding['cash'][day - 1]]
                for i in range(stockCount):
                    stock_value = consecutive_data.holding[f'stock{i}'][day - 1] * priceMat[day][i] * (1 - transFeeRate)
                    cash_possible_choices.append(stock_value)
                max_cash = max(cash_possible_choices)
                consecutive_data.holding['cash'].append(max_cash)
                from_index = cash_possible_choices.index(max_cash) - 1
                if from_index == -1:
                    consecutive_data.each_move['cash'].append([day, -1, -1, 0])
                else:
                    consecutive_data.each_move['cash'].append([day, from_index, -1, max_cash])
                # update stock
                for i in range(stockCount):
                    stock_possible_choices = []
                    from_cash = consecutive_data.holding['cash'][day - 1] / (priceMat[day][i] * (1 + transFeeRate))
                    stock_possible_choices.append(from_cash)
                    for j in range(stockCount):
                        if j == i:
                            stock_possible_choices.append(consecutive_data.holding[f'stock{i}'][day - 1])
                        else:
                            stock_j_value = consecutive_data.holding[f'stock{j}'][day - 1] * priceMat[day][j] * (1 - transFeeRate)
                            to_stock_i = stock_j_value / (priceMat[day][i] * (1 + transFeeRate))
                            stock_possible_choices.append(to_stock_i)
                    max_stock = max(stock_possible_choices)
                    consecutive_data.holding[f'stock{i}'].append(max_stock)
                    from_index = stock_possible_choices.index(max_stock)
                    if from_index == i + 1:
                        consecutive_data.each_move[f'stock{i}'].append([day, i, i, 0])
                    else:
                        from_stock = -1 if from_index == 0 else from_index - 1
                        action_amount = max_stock * priceMat[day][i]
                        consecutive_data.each_move[f'stock{i}'].append([day, from_stock, i, action_amount])
            consecutive_data.take_back_cash(first_stage_end_time)

        # stage 2
        if first_stage_end_time != 0:
            second_stage_initial_cash = consecutive_data.holding['cash'][-1]
        else:
            second_stage_initial_cash = initial_cash
        consecutive_data.sperate_cash(second_stage_initial_cash, second_stage_start_time)
        for day in range(second_stage_start_time + 1, dataLen - 1):
            cash_possible_choices = [consecutive_data.holding['cash'][day - K - 1]]
            for i in range(stockCount):
                stock_value = consecutive_data.holding[f'stock{i}'][day - K - 1] * priceMat[day][i] * (1 - transFeeRate)
                cash_possible_choices.append(stock_value)
            max_cash = max(cash_possible_choices)
            consecutive_data.holding['cash'].append(max_cash)
            from_index = cash_possible_choices.index(max_cash) - 1
            if from_index == -1:
                consecutive_data.each_move['cash'].append([day, -1, -1, 0])
            else:
                consecutive_data.each_move['cash'].append([day, from_index, -1, max_cash])
            for i in range(stockCount):
                stock_possible_choices = []
                from_cash = consecutive_data.holding['cash'][day - K - 1] / (priceMat[day][i] * (1 + transFeeRate))
                stock_possible_choices.append(from_cash)
                for j in range(stockCount):
                    if j == i:
                        stock_possible_choices.append(consecutive_data.holding[f'stock{i}'][day - K - 1])
                    else:
                        stock_j_value = consecutive_data.holding[f'stock{j}'][day - K - 1] * priceMat[day][j] * (1 - transFeeRate)
                        to_stock_i = stock_j_value / (priceMat[day][i] * (1 + transFeeRate))
                        stock_possible_choices.append(to_stock_i)
                max_stock = max(stock_possible_choices)
                consecutive_data.holding[f'stock{i}'].append(max_stock)
                from_index = stock_possible_choices.index(max_stock)
                if from_index == i + 1:
                    consecutive_data.each_move[f'stock{i}'].append([day, i, i, 0])
                else:
                    from_stock = -1 if from_index == 0 else from_index - 1
                    action_amount = max_stock * priceMat[day][i]
                    consecutive_data.each_move[f'stock{i}'].append([day, from_stock, i, action_amount])
        consecutive_data.take_back_cash(dataLen - 1)
        consecutive_data.final_income = consecutive_data.holding["cash"][-1]

        # max
        if consecutive_data.final_income > max_final_income:
            max_final_income = consecutive_data.final_income
            max_final_income_obj = consecutive_data

        first_stage_end_time += 1
        second_stage_start_time += 1

    # construct actionMat
    action = max_final_income_obj.each_move["cash"][-1]
    if action[-1] != 0:
        actionMat.append(action)
    while True:
        previous_day = action[0] - 1
        previous_from = action[1]
        if previous_from == -1:
            finding_hold = max_final_income_obj.each_move["cash"]
        else:
            finding_hold = max_final_income_obj.each_move[f"stock{previous_from}"]
        previous_action = [item for item in finding_hold if item[0] == previous_day]
        if not previous_action:
            if action[1] == -1:
                find_tack_back_day = [item for item in max_final_income_obj.each_move["cash"] if item[0] == previous_day - K]
                if not find_tack_back_day:
                    break
                else:
                    action = find_tack_back_day[0]
        else:
            action = previous_action[0]
        if action[-1] != 0:
            actionMat.append(action)
    actionMat.reverse()

    return actionMat


