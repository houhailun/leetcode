#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/8 18:01
@Author  : Hou hailun
@File    : leetcode245_最短单词距离III.py
"""

print(__doc__)

"""
给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
说明：解集不能包含重复的子集。
示例:
输入: nums = [1,2,3]
输出:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
"""


class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # 思路: 从长度为1的子集取到长度为len(nums)的nums本身
        if nums is None:
            return nums
        nums_len = len(nums)
        ret = []
        ret.extend([[i] for i in nums])
        ret.extend(nums)
        for i in range(2, nums_len):
            for j in range(nums_len-i+1):

