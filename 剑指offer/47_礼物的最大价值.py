#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution(object):
    def maxValue(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 动态规划
        # 状态转移方程: dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        # 初始: dp[0][0] = grid[0][0]
        if not grid:
            return None
        m, n = len(grid), len(grid[0])
        dp = [[None] * n for i in range(m)]
        dp[0][0] = grid[0][0]
        # 初始化第一行
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        # 初始化第一列
        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        # 双重循环
        i = j = 0
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[m-1][n-1]

obj = Solution()
print(obj.maxValue([[1,3,1],[1,5,1],[4,2,1]]))
