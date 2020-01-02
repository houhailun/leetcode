#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/28 15:48
# Author: Hou hailun

"""
给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
示例:
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
"""


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        print(self.maxSubArray_v1(nums))
        print(self.maxSubArray_greed(nums))
        print(self.maxSubArray_dp(nums))


    def maxSubArray_v1(self, nums):
        # 思路：
        # case1：当前和为负数，则再加后面的数字也必然比当前的数字小，所以直接把当前的数字认为新的起始问题
        # case2：当前和为正数，则加上后面的数字
        # case3：若当前和大于最大和，则更新最大和
        cur_sum = 0
        max_sum = nums[0]
        for i in nums:
            if cur_sum < 0:  # 当前和小于0，则直接取i
                cur_sum = i
            else:
                cur_sum += i
            if cur_sum > max_sum:
                max_sum = cur_sum
        return max_sum

    def maxSubArray_greed(self, nums):
        # 贪心算法
        # 使用单个数组作为输入来查找最大（或最小）元素（或总和）的问题，贪心算法是可以在线性时间解决的方法之一。
        # 每一步都选择最佳方案，到最后就是全局最优的方案

        # 需要在遍历数组的时候维护3个变量：当前和，最大和，当前元素
        n = len(nums)
        cur_sum = max_sum = nums[0]
        for i in range(1, n):
            cur_sum = max(nums[i], cur_sum+nums[i])
            max_sum = max(max_sum, cur_sum)
        return max_sum

    def maxSubArray_dp(self, nums):
        # 动态规划法
        # 在整个数组或在固定大小的滑动窗口中找到总和或最大值或最小值的问题可以通过动态规划（DP）在线性时间内解决。
        # 有两种标准DP方法适用于数组：
        #     常数空间，沿数组移动并在原数组修改。
        #     线性空间，首先沿left->right方向移动，然后再沿right->left方向移动。 合并结果。
        # 我们在这里使用第一种方法，因为可以修改数组跟踪当前位置的最大和。
        # 下一步是在知道当前位置的最大和后更新全局最大和
        n = len(nums)
        max_sum = nums[0]
        for i in range(1, n):
            # i-1的和大于0则累加到i；否则i不变
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            max_sum = max(nums[i], max_sum)
        return max_sum



obj = Solution()
nums = [-2,1,-3,4,-1,2,1,-5,4]
print(obj.maxSubArray(nums))