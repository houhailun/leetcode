#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/8 17:33
@Author  : Hou hailun
@File    : leetcode268_缺失数字.py
"""

print(__doc__)

"""
给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
示例 1:
输入: [3,0,1]
输出: 2
示例 2:
输入: [9,6,4,2,3,5,7,0,1]
输出: 8
"""


class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1：排序后查找nums[i] == i
        # nums.sort()
        # for i in range(len(nums)):
        #     if i != nums[i]:
        #         return i
        # return -1

        # 方法2：求和后在相减
        sum = len(nums)
        for i in range(len(nums)):
            sum += i
            print(sum)
            sum -= nums[i]
            print(sum)
        return sum


obj = Solution()
ret = obj.missingNumber([3,1,0])
print(ret)