#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/23 10:24
# Author: Hou hailun

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

# 思路：初始子集只有[], 从前往后遍历，遇到一个数就把所有子集加上该数组成新的子集，遍历完毕即是所有子集
# [] -> [],[1] -> [],[1],[2],[1,2] -> [],[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]
import pysnooper


class Solution:
    # @pysnooper.snoop()
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        if nums is None:
            return res

        res.append([])
        for num in nums:
            new = [s + [num] for s in res]
            res = res + new
        return res


obj = Solution()
ret = obj.subsets([1, 2, 3])
print(ret)
