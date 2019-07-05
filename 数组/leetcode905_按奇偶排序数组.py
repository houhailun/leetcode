#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 15:31
@Author  : Hou hailun
@File    : leetcode905_按奇偶排序数组.py
"""

print(__doc__)

"""
给定一个非负整数数组 A，返回一个数组，在该数组中， A 的所有偶数元素之后跟着所有奇数元素。
你可以返回满足此条件的任何数组作为答案。
示例：
输入：[3,1,2,4]
输出：[2,4,3,1]
输出 [4,2,3,1]，[2,4,1,3] 和 [4,2,1,3] 也会被接受。

提示：
1 <= A.length <= 5000
0 <= A[i] <= 5000
"""


class Solution(object):
    def sortArrayByParity(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        return sorted(A, key=lambda x: x % 2)


obj = Solution()
ret = obj.sortArrayByParity([3,1,2,4])
print(ret)