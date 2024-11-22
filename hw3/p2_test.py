import numpy as np

def stock_trading_distributed_cash(priceMat, transFeeRate, K):
    prices = np.array(priceMat)
    s, a = prices.shape  # s: 交易天数，a: 股票数量
    initial_cash = 1000  # 初始资金

    max_cash_days = K
    max_total_cash_days = K

    # dp[day][cash_days][holding] = (资金量, 前一天的状态, 动作)
    dp = np.full((s + 1, K + 1, a + 1), -np.inf, dtype=object)
    for day in range(s + 1):
        for cash_days in range(K + 1):
            for holding in range(a + 1):
                dp[day][cash_days][holding] = (-np.inf, None, None)

    # 初始状态：第 0 天，持有现金，cash_days = 0
    dp[0][0][a] = (initial_cash, None, None)  # holding = a 表示持有现金

    # 动态规划过程
    for day in range(1, s + 1):
        for cash_days in range(K + 1):
            for holding in range(a + 1):
                current_value, prev_state, action = dp[day - 1][cash_days][holding]
                if current_value == -np.inf:
                    continue

                # 持有现金的状态
                if holding == a:
                    # 继续持有现金
                    if cash_days + 1 <= K:
                        new_cash_days = cash_days + 1
                        new_value = current_value
                        if dp[day][new_cash_days][a][0] < new_value:
                            dp[day][new_cash_days][a] = (new_value, (cash_days, holding), None)
                    else:
                        # 超过最大持有现金天数，不能继续持有现金
                        continue

                    # 购买股票
                    for j in range(a):
                        stock_price = prices[day - 1][j]
                        if stock_price <= 0:
                            continue
                        new_value = (current_value * (1 - transFeeRate)) / stock_price
                        if dp[day][cash_days][j][0] < new_value:
                            dp[day][cash_days][j] = (new_value, (cash_days, holding), (day - 1, -1, j, current_value * (1 - transFeeRate)))
                else:
                    # 持有股票的状态
                    stock_price = prices[day - 1][holding]
                    # 继续持有股票
                    new_value = current_value
                    if dp[day][cash_days][holding][0] < new_value:
                        dp[day][cash_days][holding] = (new_value, (cash_days, holding), None)

                    # 卖出股票
                    if cash_days + 1 <= K:
                        new_cash_days = cash_days + 1
                        sell_cash = current_value * stock_price * (1 - transFeeRate)
                        if dp[day][new_cash_days][a][0] < sell_cash:
                            dp[day][new_cash_days][a] = (sell_cash, (cash_days, holding), (day - 1, holding, -1, sell_cash))
                    else:
                        # 超过最大持有现金天数，不能再持有现金
                        continue

                    # 卖出股票 i，购买股票 j
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
    # 寻找在第 s 天，cash_days >= K，资金量最大的状态
    max_value = -np.inf
    end_state = None
    for cash_days in range(K, K + 1):
        for holding in range(a + 1):
            current_value, _, _ = dp[s][cash_days][holding]
            if current_value > max_value:
                max_value = current_value
                end_state = (s, cash_days, holding)

    if end_state is None:
        return [], 0  # 无法满足条件

    # 如果持有股票，需要在最后一天卖出
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
            # 无法在最后一天卖出股票并持有现金，因为会超过最大持有现金天数
            return [], 0

    # 回溯路径
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
    return actions, max_value

# 示例用法
priceMat = [
    [2, 3, 1, 4],  # Day 1 prices for stocks 0 to 3
    [2, 2, 2, 5],
    [3, 1, 3, 6],
    [4, 3, 4, 5],
    [5, 4, 5, 4]
]
transFeeRate = 0.01  # 交易费用率
K = 2  # 至少持有现金的天数

actions, max_wealth = stock_trading_distributed_cash(priceMat, transFeeRate, K)

print("最大资金量：", max_wealth)
print("交易记录（actionMat）：")
for action in actions:
    print(action)