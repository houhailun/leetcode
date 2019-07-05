#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/5 17:03
@Author  : Hou hailun
@File    : leetcode169_求众数.py
"""

print(__doc__)

"""
给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在众数。
示例 1:
输入: [3,2,3]
输出: 3
示例 2:
输入: [2,2,1,1,1,2,2]
输出: 2
"""


class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1: 排序后去中间的数字
        # 利用快排核心Partition函数，返回index=n/2
        # 利用出现次数超过一半特性
        # cnt = 1
        # ret = nums[0]
        # for i in range(1, len(nums)):
        #     if cnt <= 0:
        #         ret = nums[i]
        #     if nums[i] == ret:
        #         cnt += 1
        #     else:
        #         cnt -= 1
        # return ret

        # Counter
        from collections import Counter
        c = Counter(nums)
        return c.most_common(1)[0][0]

nums = [2,2,1,1,1,2,2]
obj = Solution()
ret = obj.majorityElement(nums)
print(ret)