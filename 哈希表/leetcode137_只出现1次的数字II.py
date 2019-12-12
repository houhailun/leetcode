#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/12 18:33
# Author: Hou hailun

"""
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现了三次。找出那个只出现了一次的元素。
示例 1:
输入: [2,2,3,2]
输出: 3
示例 2:
输入: [0,1,0,1,0,1,99]
输出: 99
"""


class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1：数学公式
        x = set(nums)
        return (3*sum(x) - sum(nums)) / 2