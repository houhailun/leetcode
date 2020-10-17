#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 方法：穷举，两次for循环，每次找到比当前值大的最大值，这样可以得到在当前下最大利润，并更新最大利润
        # 时间复杂度为O(N*N), N较大时会导致超出时间限制
        # if not prices and len(prices) == 1:
        #     return 0
        # max_profit = cur_max = 0
        # len_price = len(prices)
        # for i in range(len_price-1):
        #     for j in range(i+1, len_price):
        #         if prices[j] > prices[i]:
        #             cur_max = max(cur_max, prices[j] - prices[i])
        #     max_profit = max(max_profit, cur_max)
        # return max_profit

        # 方法2：动态规划法
        # 定义状态: dp[i] 表示前i日的最大利润
        # 状态转移方程: dp[i] = max(dp[i-1], prices[i]-min(price[0:i]))
        # 初始: dp[0] = 0, 表示第一天利润为0
        # 返回值dp[n-1]
        # _len = len(prices)
        # dp = [0] * (_len+1)
        # dp[0] = 0
        # for i in range(1, _len):
        #     dp[i] = max(dp[i-1], prices[i] - min(prices[:i]))
        # print(dp)
        # return dp[_len-1]

        # 方法3: 一次遍历，记录最小价格，当前最大利润
        # 24ms
        min_price = float('inf')
        max_profit = 0
        for price in prices:
            if price < min_price:
                min_price = price
            if max_profit < price - min_price:
                max_profit = price - min_price
        return max_profit


obj = Solution()
print(obj.maxProfit([7,1,5,3,6,4]))