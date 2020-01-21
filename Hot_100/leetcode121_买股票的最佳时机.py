#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/19 14:09
# Author: Hou hailun

"""
给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
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
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0

        # 方法1：暴力破解
        # return self.maxProfit_v1(prices)

        # 方法2：
        # min_price: 迄今为止所得到的最小的谷值
        # max_profit: 迄今为止最大的利润（卖出价格与最低价格之间的最大差值)
        min_price = 999999
        max_profit = 0
        for price in prices:
            if price < min_price:
                min_price = price
            elif max_profit < price-min_price:
                max_profit = price-min_price
        return max_profit

    def maxProfit_v1(self, prices):
        # 在需要找出给定数组中两个数字之间的最大差值（即，最大利润）。此外，第二个数字（卖出价格）必须大于第一个数字
        cnt = len(prices)
        max_profits = 0
        for i in range(cnt-1):
            for j in range(i+1, cnt):
                profits = prices[j] - prices[i]
                if profits > max_profits:
                    max_profits = profits
        return max_profits