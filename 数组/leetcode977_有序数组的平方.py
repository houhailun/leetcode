#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 15:11
@Author  : Hou hailun
@File    : leetcode977_有序数组的平方.py
"""

print(__doc__)

"""
给定一个按非递减顺序排序的整数数组 A，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。
示例 1：
输入：[-4,-1,0,3,10]
输出：[0,1,9,16,100]
示例 2：
输入：[-7,-3,2,3,11]
输出：[4,9,9,49,121]
提示：
1 <= A.length <= 10000
-10000 <= A[i] <= 10000
A 已按非递减顺序排序。
"""

class Solution(object):
    def sortedSquares(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        # 方法1：求平方后sort
        # for i, x in enumerate(A):
        #     A[i] = x ** 2
        # A.sort()
        # return A

        # 方法2
        return sorted([i*i for i in A])


a = [-4,-1,0,3,10]
obj = Solution()
ret = obj.sortedSquares(a)
print(a)
print(ret)