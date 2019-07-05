#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。
说明：不要使用任何内置的库函数，如  sqrt。
示例 1：
输入：16
输出：True
示例 2：
输入：14
输出：False
"""


class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        # 二分
        start, end = 0, num
        while start <= end:
            mid = (start + end) >> 1
            if mid ** 2 == num:
                return True
            elif mid ** 2 < num:
                start = mid + 1
            else:
                end = mid - 1
        return False


cls = Solution()
print(cls.isPerfectSquare(16))