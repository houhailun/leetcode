#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述：给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。

说明:
返回的下标值（index1 和 index2）不是从零开始的。
你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
示例:
输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

解题思路：
    1、first=num[0],end=num[-1]
    2、如果first + end < target -> first += 1
    3、如果first + end > target -> end -= 1
"""


class Solution:
    def two_sum(self, numbers, target):
        if not numbers or len(numbers) < 2:
            return None, None
        first, end = 0, len(numbers)-1
        while first < end:
            if numbers[first] + numbers[end] == target:
                return first+1, end+1
            elif numbers[first] + numbers[end] > target:
                end -= 1
            else:
                first += 1
        return None, None


cls = Solution()
print(cls.two_sum([2, 7, 11, 15], 18))