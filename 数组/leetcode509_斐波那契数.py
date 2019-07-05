#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 15:52
@Author  : Hou hailun
@File    : leetcode509_斐波那契数.py
"""

print(__doc__)

"""
斐波那契数，通常用 F(n) 表示，形成的序列称为斐波那契数列。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
给定 N，计算 F(N)。
示例 1：
输入：2
输出：1
解释：F(2) = F(1) + F(0) = 1 + 0 = 1.
示例 2：
输入：3
输出：2
解释：F(3) = F(2) + F(1) = 1 + 1 = 2.
示例 3：
输入：4
输出：3
解释：F(4) = F(3) + F(2) = 2 + 1 = 3.
"""


class Solution(object):
    def fib(self, N):
        """
        :type N: int
        :rtype: int
        """
        n, a, b = 0, 0, 1
        while n < N:
            a, b = b, a+b
            n = n + 1
        return a


obj = Solution()
ret = obj.fib(4)
print(ret)