#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 有序数组 -> 二分
        # 数字范围是0~n-1,若没有缺失，则必然是数i位于第i个，因此只需要二分判断mid == nums[mid]
        if not nums:
            return None
        start, end = 0, len(nums)-1
        while start <= end:
            mid = (start + end) // 2
            if mid == nums[mid]:  # 左边的必然相等，右边可能存在不等
                start = mid + 1
            else:  # 右边的必然不等，左边可能存在不等
                end = mid - 1
        return start