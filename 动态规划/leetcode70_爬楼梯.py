
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/28 15:26
# Author: Hou hailun


class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        self.climbStairs_fib(n)  # 斐波那契法

        self.climbStairs_dp(n)   # 动态规划法

    def climbStairs_fib(self, n):
        if n <= 0:
            return 0
        a, b = 0, 1
        for i in range(n):
            a, b = b, a+b
        return b

    def climbStairs_dp(self, n):
        # 第i阶可以由以下两种方法得到：
        #     在第(i - 1) 阶后向上爬一阶。
        #     在第(i - 2) 阶后向上爬2阶。
        # 所以到达第i阶的方法总数就是到第(i−1) 阶和第(i−2) 阶的方法数之和。
        # 令dp[i]表示能到达第i阶的方法总数：
        #     dp[i] = dp[i−1]+dp[i−2]
        if n == 1:
            return 1
        dp = [None] * (n+1)
        dp[1], dp[2] = 1, 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]