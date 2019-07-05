#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目名称：搜索插入位置

题目描述：给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。你可以假设数组中无重复元素。
示例 1:
输入: [1,3,5,6], 5
输出: 2
示例 2:
输入: [1,3,5,6], 2
输出: 1
示例 3:
输入: [1,3,5,6], 7
输出: 4
示例 4:
输入: [1,3,5,6], 0
输出: 0
"""

class Solution:
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        '''
        方法1：利用python的index,append,sort函数，时间复杂度为O(NlogN)
        if target in nums:
            return nums.index(target)
        nums.append(target)
        nums.sort()
        return nums.index(target)
        '''
        # 方法2：排序数据利用二分法查找target，若找到则返回索引，找不到则遍历数据，找到对应位置后插入
        len_num = len(nums)-1
        start, end = 0, len_num
        mid = (start + end) // 2
        while start < end:
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid
            else:
                start = mid + 1
            mid = (start + end) // 2

        # target 不在nums中,必然start = end = len(nums) 或者 = 0
        if nums[mid] >= target:
            return mid
        return mid+1


cls = Solution()
print(cls.searchInsert([1,3], 0))