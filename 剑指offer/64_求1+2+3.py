#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def sumNums(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = 0

        # 方法1: 递归
        # return n and (n + self.sumNums(n-1))

        # 方法2：python的sum
        # return sum(range(n+1))

        # 方法3：算数法
        return (n + 1) * n >> 1