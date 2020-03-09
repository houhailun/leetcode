#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 16:04
# Author: Hou hailun

# 一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：
# F(0) = 0,   F(1) = 1
# F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
# 斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。
#
# 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。


class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 方法1：暴力法
        # print(self.fib_helper1(n))

        # 方法2：动态规划，记录中间计算结果
        print(self.fib_dp(n) % 1000000007)

        # 方法3
        return self.fib_helper3(n) % 1000000007

    def fib_helper3(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        a, b = 0, 1
        for i in range(n):
            a, b = b, a+b
        return a

    def fib_dp(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1
        dp = [0] * (n+1)
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    def fib_helper1(self, n):
        if n == 0:
            return 0
        if n == 1:
            return 1

        res = self.fib_helper1(n-1) + self.fib_helper1(n-2)
        return res

obj = Solution()
print(obj.fib(45))