#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 15:39
@Author  : Hou hailun
@File    : leetcode561_数组拆分.py
"""

print(__doc__)

"""
给定长度为 2n 的数组, 你的任务是将这些数分成 n 对, 例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从1 到 n 的 min(ai, bi) 总和最大。
示例 1:
输入: [1,4,3,2]
输出: 4
解释: n 等于 2, 最大总和为 4 = min(1, 2) + min(3, 4)
"""


class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1：排序后相邻2个元素作为1对,第1个数小，全部累加即可
        # nums.sort()
        # ret = 0
        # for i in range(0, len(nums), 2):
        #     ret += nums[i]
        # return ret

        # 方法2：一行代码实现
        return sum(sorted(nums)[::2])


obj = Solution()
ret = obj.arrayPairSum([1,4,3,2])
print(ret)