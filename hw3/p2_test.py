import numpy as np

def stock_trading(prices, k, rho):
    s = len(prices)          # 交易天数
    a = len(prices[0])       # 股票数量
    initial_cash = 1.0       # 初始资金
    max_cash_days = k        # 至少持有现金的天数

    # 将价格数组转换为 NumPy 数组，便于计算
    prices = np.array(prices)

    # dp_value 存储在每个状态下的最大资金量
    dp_value = np.full((s + 1, max_cash_days + 1, a + 1), -np.inf)
    # dp_prev 存储前一个状态的索引，用于回溯路径
    dp_prev = np.empty((s + 1, max_cash_days + 1, a + 1), dtype=object)
    # dp_action 存储达到当前状态的操作
    dp_action = np.empty((s + 1, max_cash_days + 1, a + 1), dtype=object)

    # 初始状态：第 0 天，持有现金，剩余需要持有现金的天数为 k
    dp_value[0][k][0] = initial_cash

    # 动态规划过程
    for day in range(1, s + 1):
        for remaining_cash_days in range(max_cash_days + 1):
            for holding in range(a + 1):
                # 检查状态是否可达
                if dp_value[day - 1][remaining_cash_days][holding] == -np.inf:
                    continue

                current_value = dp_value[day - 1][remaining_cash_days][holding]

                # 计算剩余天数，确保 remaining_cash_days 不超过剩余天数
                remaining_days = s - day + 1
                if remaining_cash_days > remaining_days:
                    continue

                # 持有现金的状态
                if holding == 0:
                    # 继续持有现金
                    new_remaining_cash_days = max(0, remaining_cash_days - 1)
                    if current_value > dp_value[day][new_remaining_cash_days][0]:
                        dp_value[day][new_remaining_cash_days][0] = current_value
                        dp_prev[day][new_remaining_cash_days][0] = (remaining_cash_days, 0)
                        dp_action[day][new_remaining_cash_days][0] = None  # 无操作

                    # 购买股票 j
                    for j in range(1, a + 1):
                        stock_price = prices[day - 1][j - 1]
                        if stock_price <= 0:
                            continue  # 跳过价格为 0 的股票

                        # 计算购买后的资金量（以股票数量表示）
                        new_value = (current_value * (1 - rho)) / stock_price

                        if new_value > dp_value[day][remaining_cash_days][j]:
                            dp_value[day][remaining_cash_days][j] = new_value
                            dp_prev[day][remaining_cash_days][j] = (remaining_cash_days, 0)
                            dp_action[day][remaining_cash_days][j] = [day - 1, -1, j - 1, current_value * (1 - rho)]
                else:
                    # 持有股票，股票索引为 holding - 1
                    stock_index = holding - 1
                    stock_price = prices[day - 1][stock_index]

                    # 继续持有股票
                    if current_value > dp_value[day][remaining_cash_days][holding]:
                        dp_value[day][remaining_cash_days][holding] = current_value
                        dp_prev[day][remaining_cash_days][holding] = (remaining_cash_days, holding)
                        dp_action[day][remaining_cash_days][holding] = None  # 无操作

                    # 卖出股票，转换为现金
                    new_remaining_cash_days = max(0, remaining_cash_days - 1)
                    sell_cash = current_value * stock_price * (1 - rho)
                    if sell_cash > dp_value[day][new_remaining_cash_days][0]:
                        dp_value[day][new_remaining_cash_days][0] = sell_cash
                        dp_prev[day][new_remaining_cash_days][0] = (remaining_cash_days, holding)
                        dp_action[day][new_remaining_cash_days][0] = [day - 1, stock_index, -1, sell_cash]

                    # 卖出股票 i，购买股票 j
                    for j in range(1, a + 1):
                        if j == holding:
                            continue  # 不购买同一支股票
                        new_stock_price = prices[day - 1][j - 1]
                        if new_stock_price <= 0:
                            continue  # 跳过价格为 0 的股票

                        # 卖出股票 i，购买股票 j
                        switch_cash = current_value * stock_price * (1 - rho)
                        new_value = (switch_cash * (1 - rho)) / new_stock_price

                        if new_value > dp_value[day][remaining_cash_days][j]:
                            dp_value[day][remaining_cash_days][j] = new_value
                            dp_prev[day][remaining_cash_days][j] = (remaining_cash_days, holding)
                            dp_action[day][remaining_cash_days][j] = [day - 1, stock_index, j - 1, switch_cash]
    
    # 回溯找到最大资金量的路径
    max_value = -np.inf
    end_state = None
    for holding in range(a + 1):
        if dp_value[s][0][holding] > max_value:
            max_value = dp_value[s][0][holding]
            end_state = (s, 0, holding)

    # 如果持有股票，需要在最后一天卖出
    if end_state[2] != 0:
        holding = end_state[2]
        stock_index = holding - 1
        stock_price = prices[s - 1][stock_index]
        final_cash = dp_value[s][0][holding] * stock_price * (1 - rho)
        dp_value[s][0][0] = final_cash
        dp_prev[s][0][0] = (0, holding)
        dp_action[s][0][0] = [s - 1, stock_index, -1, final_cash]
        end_state = (s, 0, 0)
        max_value = final_cash

    # 构建 actionMat
    actionMat = []
    state = end_state
    while state[0] > 0:
        day, remaining_cash_days, holding = state
        action = dp_action[day][remaining_cash_days][holding]
        prev_remaining_cash_days, prev_holding = dp_prev[day][remaining_cash_days][holding]

        if action is not None:
            actionMat.append(action)

        state = (day - 1, prev_remaining_cash_days, prev_holding)

    # 由于是倒序回溯，需要将 actionMat 反转
    actionMat.reverse()

    return max_value, actionMat

# 示例用法
prices = [
    [2, 3, 1, 4],  # Day 1 prices for stocks 0 to 3
    [2, 2, 2, 5],
    [3, 1, 3, 6],
    [4, 3, 4, 5],
    [5, 4, 5, 4]
]
k = 2          # 至少持有现金的天数
rho = 0.01     # 交易费用率

max_wealth, actionMat = stock_trading(prices, k, rho)

print("最大资金量：", max_wealth)
print("交易记录（actionMat）：")
for action in actionMat:
    print(action)
