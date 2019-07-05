#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 15:57
@Author  : Hou hailun
@File    : leetcode867_转置矩阵.py
"""

print(__doc__)

"""
给定一个矩阵 A， 返回 A 的转置矩阵。
矩阵的转置是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。
示例 1：
输入：[[1,2,3],[4,5,6],[7,8,9]]
输出：[[1,4,7],[2,5,8],[3,6,9]]
示例 2：
输入：[[1,2,3],[4,5,6]]
输出：[[1,4],[2,5],[3,6]]
"""


class Solution(object):
    def transpose(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        # 方法1: 利用numpy实现(执行勇士200ms)
        # import numpy as np
        # mat_a = np.array(A)
        # return mat_a.T

        # 方法2：
        return [[row[i] for row in A] for i in range(len(A[0]))]


obj = Solution()
ret = obj.transpose([[1,2,3],[4,5,6],[7,8,9]])
print(ret)