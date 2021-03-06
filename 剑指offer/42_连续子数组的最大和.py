#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 思路:max_sum标记连续子数组的最大和
        # 若当前和小于等于0，则说明加上后面的值X只会比后面的X小，因此丢弃当前和，当前和=X
        # 若当前和大于max_sum, 更新
        if not nums:
            return None
        cur_sum = max_sum = float("-inf")
        for num in nums:
            if cur_sum <= 0:
                cur_sum = num
            else:
                cur_sum += num
            if cur_sum > max_sum:
                max_sum = cur_sum
        return max_sum

    def maxSubArrayAndIndex(self, nums):
        # 同时输出子数组的索引范围
        cur_sum = max_sum = float('-inf')
        start = end = 0
        for ix, num in enumerate(nums):
            if cur_sum < 0:
                cur_sum = num
                start = ix  # 指定最大子数组的开始索引
            else:
                cur_sum += num

            if cur_sum > max_sum:
                max_sum = cur_sum
                end = ix  # 指定最大子数组的结束索引
        return max_sum, start, end

obj = Solution()
nums = [1, 2, -4, 3, 5, -4, 8, -9, 10]
print(obj.maxSubArray(nums))
print(obj.maxSubArrayAndIndex(nums))