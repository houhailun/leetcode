#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 10:45
# Author: Hou hailun

"""
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。
"""


class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 斐波那契数列
        # if n == 0:
        #     return 0
        # if n == 1:
        #     return 1
        # if n == 2:
        #     return 2

        # a, b = 1, 2
        # for i in range(3, n+1):
        #     tmp = a + b
        #     a = b
        #     b = tmp
        # return b

        # 动态规划法
        if n == 1:
            return 1
        dp = [None] * (n + 1)
        dp[1], dp[2] = 1, 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]