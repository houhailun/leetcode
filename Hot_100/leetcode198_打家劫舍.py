#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 13:19
# Author: Hou hailun

# 思路：本题实际上就是动态规划问题  dp[i] = max(dp[i-2] + nums[i], dp[i-1])
# dp[i]表示从前i间房子所能抢到的最大金额, Ai表示第i间房子的金额
# i=1，只有1间房子，则dp[1]=A1
# i=2, 有2间房子，则dp[2] = max(A1，A2)
# i=3，有3间房子，则dp[3]有两种情况：1、抢第三间，数值相加  2、不枪第三间，保持当前数值；dp[3]=max(dp[2], dp[1]+A3)
# ...
# i=k, 有k间房子，则同样可得dp[k]=max(dp[k-1], Ak+dp[k-2])


class Solution:
    def rob(self, nums):
        if not nums:
            return 0
        if len(nums) < 3:
            return max(nums)
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        return dp[-1]

