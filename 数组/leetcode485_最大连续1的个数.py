#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/8 17:26
@Author  : Hou hailun
@File    : leetcode485_最大连续1的个数.py
"""

print(__doc__)

"""
输入: [1,1,0,1,1,1]
输出: 3
解释: 开头的两位和最后的三位都是连续1，所以最大连续1的个数是 3.
"""


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_ones = cur_ones = 0
        for num in nums:
            if num == 1:
                cur_ones += 1
            else:
                cur_ones = 0
            if cur_ones > max_ones:
                max_ones = cur_ones
        return max_ones


obj = Solution()
ret = obj.findMaxConsecutiveOnes([1,1,0,1,1,1])
print(ret)
