#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 15:20
@Author  : Hou hailun
@File    : leetcode1051_高度检查器.py
"""

print(__doc__)

"""
学校在拍年度纪念照时，一般要求学生按照 非递减 的高度顺序排列。
请你返回至少有多少个学生没有站在正确位置数量。该人数指的是：能让所有学生以 非递减 高度排列的必要移动人数。
示例：
输入：[1,1,4,2,1,3]
输出：3
解释：
高度为 4、3 和最后一个 1 的学生，没有站在正确的位置。

提示：
1 <= heights.length <= 100
1 <= heights[i] <= 100
"""

class Solution(object):
    def heightChecker(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # 其实就是比较与排序数据有多少位不匹配
        tmp_arr = sorted(heights)
        cnt = 0
        # for x, y in zip(heights, tmp_arr):
        #     if x != y:
        #         cnt += 1

        for i in range(len(heights)):
            if heights[i] != tmp_arr[i]:
                cnt += 1
        return cnt

obj = Solution()
ret = obj.heightChecker([1,1,4,2,1,3])
print(ret)