#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/28 16:16
# Author: Hou hailun

"""
给定一个整数数组  nums，求出数组从索引 i 到 j  (i ≤ j) 范围内元素的总和，包含 i,  j 两点。
示例：
给定 nums = [-2, 0, 3, -5, 2, -1]，求和函数为 sumRange()
sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
"""


class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """

        # python内置函数, 时间复杂度O(N), 空间复杂度O(1)
        return sum(self.nums[i:j+1])


class NumArray_v2(object):
    # 第一种方法中，每次查询时间复杂度为O(N), 多次查询比较浪费时间；考虑在初始化的时候做工作，保障查询只需要O(1)即可
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        if not nums:
            return

        n = len(nums)
        self.dp = [0] * (n+1)
        self.dp[1] = nums[0]
        # 遍历数组，保存累加和
        for i in range(2, n+1):
            self.dp[i] = nums[i-1] + self.dp[i-1]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.dp[j+1] - self.dp[i]