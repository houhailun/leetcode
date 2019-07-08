#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/8 17:55
@Author  : Hou hailun
@File    : leetcode217_存在重复元素.py
"""

print(__doc__)

"""
给定一个整数数组，判断是否存在重复元素。
如果任何值在数组中出现至少两次，函数返回 true。如果数组中每个元素都不相同，则返回 false。
示例 1:输入: [1,2,3,1]
输出: true
示例 2:输入: [1,2,3,4]
输出: false
示例 3:输入: [1,1,1,3,3,4,3,2,4,2]
输出: true
"""


class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 方法1: 使用字典构建哈希表
        occur_cnt = {}
        for num in nums:
            if num in occur_cnt:
                return True
            occur_cnt[num] = 0

        return False

        # 方法2：集合去重
        return False if len(nums) == len(set(nums)) else True


obj = Solution()
ret = obj.containsDuplicate([1,2,3,4])
print(ret)