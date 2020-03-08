#!/usr/bin/env python
# -*- encoding:utf-8 -*-


"""
假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素。
你的算法时间复杂度必须是 O(log n) 级别。
示例 1:
输入: nums = [4,5,6,7,0,1,2], target = 0
输出: 4
示例 2:
输入: nums = [4,5,6,7,0,1,2], target = 3
输出: -1
"""


class Solution:
    def search(self, nums, target):
        # # 方法1：利用index函数
        # try:
        #     return nums.index(target)
        # except:
        #     return -1

        # 方法2：由于数组是基本有序，在有序或者基本有序数组中查找某个值一般使用二分查找
        # 本题难点在如何确认当前值是上一个有序数组还是下一个有序数组
        # 1、当前值大于前面的元素，则属于上一个有序数组
        # 2、当前值小于后面的元素，则属于下一个有序数组
        low, high = 0, len(nums) - 1
        while low <= high:
            mid = low + (high - low) // 2

            if nums[mid] == target:
                return mid
            # 右半边有序
            elif nums[mid] < nums[high]:
                if nums[mid] < target <= nums[high]:  # 右边查找
                    low = mid + 1
                else:
                    high = mid - 1
            else:  # 左半边有序
                if nums[low] <= target < nums[mid]:  # 左边查找
                    high = mid - 1
                else:
                    low = mid + 1

        return -1


if __name__ == "__main__":
    obj = Solution()
    print(obj.search([4,5,6,7,0,1,2], 0))
