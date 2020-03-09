#!/usr/bin/env python
# -*- encoding:utf-8 -*-


"""
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
你的算法时间复杂度必须是 O(log n) 级别。
如果数组中不存在目标值，返回 [-1, -1]。
示例 1:
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]
示例 2:
输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]
"""


class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 思路：有序数组 -- 二分法
        # 重点在于：找到target后还需要检查前面或者后面是否等于target
        if not nums:
            return [-1, -1]

        first = last = 0
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            print('low:%s,high:%s,mid:%s', low, high, mid)
            if nums[mid] == target:
                first = last = mid
                print(mid, nums[mid])
                while first >= 0 and nums[first] == target:
                    first -= 1
                while last <= len(nums) - 1 and nums[last] == target:
                    last += 1
                return [first+1, last-1]
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1

        return [-1, -1]


if __name__ == "__main__":
    obj = Solution()
    print(obj.searchRange([1], 1))