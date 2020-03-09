#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 16:17
# Author: Hou hailun

# 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
#
# 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

# 解题思路：本题仍属于斐波纳契数列


class Solution(object):
    def numWays(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2
        pre, nex = 1, 2
        for i in range(3, n+1):
            pre, nex = nex, pre+nex
        return nex % 1000000007


obj = Solution()
print(obj.numWays(7))