#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述:给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
注意你不能在买入股票前卖出股票。

示例 1:
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
示例 2:
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

解题思路:
    方法1：a、检查是否买入价格大于卖出价格（数据由大到小排序）->返回0; b、两次循环找最大利润
    方法2：动态规划、贪心
"""


class Solution:
    def max_profit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        # 问题：时间复杂度O(N*N),超出时间限制
        if not prices:
            return 0
        if prices == sorted(prices, reverse=True):
            return 0
        max_money = -1
        cur_monry = 0
        len_price = len(prices)
        for i in range(len_price):
            for j in range(i+1, len_price):
                cur_monry = prices[j] - prices[i]
                if cur_monry > 0 and cur_monry > max_money:
                    max_money = cur_monry
        return max_money

    def max_profit_v2(self, prices):
        if not prices:
            return 0

        # 一次遍历,每次找到当前最小的买入金额，并比较当前价格和最小买入金额的利润是否大于最大利润
        buy, sell = 999999, 0
        for val in prices:
            buy = min(buy, val)
            sell = max(sell, val-buy)

        return sell

cls = Solution()
print(cls.max_profit_v2([7,1,5,3,6,4]))