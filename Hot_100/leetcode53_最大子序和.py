#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 10:31
# Author: Hou hailun

"""
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
"""

# 思路：
# case1：当前和为负数，则再加后面的数字也必然比当前的数字小，所以直接把当前的数字认为新的起始问题
# case2：当前和为正数，则加上后面的数字
# case3：若当前和大于最大和，则更新最大和


class Solution:
    def findMaxSub(self, nums):
        import sys
        cur_sum = 0
        if sys.version.startswith('3'):
            max_sum = -sys.maxsize
        else:
            max_sum = -sys.maxint

        for num in nums:
            if cur_sum < 0:
                cur_sum = num
            else:
                cur_sum += num
            if cur_sum > max_sum:
                max_sum = cur_sum
        return max_sum
