#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 双指针法，p1指向开始，p2指向末尾
        # case1：p1+p1 < target: p1+=1
        # case2：p1+p2 > target: p2-=1
        res = []
        if not nums:
            return res
        start, end = 0, len(nums)-1
        while start < end:
            _sum = nums[start] + nums[end]
            if _sum < target:
                start += 1
            elif _sum > target:
                end -= 1
            else:
                return [nums[start], nums[end]]
        return res