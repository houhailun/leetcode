#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。
示例 1：
输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
示例 2：
输入: "cbbd"
输出: "bb"
"""


class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        # self.longestPalindrome_bp(s)

        self.longestPalindrome_dp(s)


    def longestPalindrome_dp(self, s):
        # 动态规划
        # dp[i][j] 表示 s[i,j]是否是回文串
        # 状态转移方程 dp[i][j] = (dp[i+1][j-1] and s[i]==s[j])
        # 初始: dp[i][i] = True
        size = len(s)
        if s < 2:
            return s

        dp = [[False for _ in range(size)] for _ in range(size)]

        max_len = 1
        start = 0
        for i in range(size):  # dp[i][i]表示s中的单个字符，是回文串
            dp[i][i] = True

        # i在j前面，j从1开始
        for j in range(1, size):
            for i in range(0, j):
                if s[i] == s[j]:
                    if j - i < 3:  # j-1 与 i+1 不构成边界(i,i+1,j-1,j)至少需要4个，举例ab, aba
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = False  # s[i] != s[j], 则s[i,j]必然不是回文串

                if dp[i][j]:
                    cur_len = j - i + 1
                    if cur_len > max_len:
                        max_len = cur_len
                        start = i
        return s[start: start+max_len]


    def longestPalindrome_bp(self, s):
        # 方法1：暴力法
        # 特判
        size = len(s)
        if size < 2:  # 单字符是回文串
            return s

        max_len = 1
        res = s[0]

        # 枚举所有长度大于等于 2 的子串
        for i in range(size - 1):
            for j in range(i + 1, size):
                if j - i + 1 > max_len and self.__valid(s, i, j):
                    max_len = j - i + 1
                    res = s[i:j + 1]
        return res


    def __valid(self, s, left, right):
        # 验证子串 s[left, right] 是否为回文串
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True


obj = Solution()
print(obj.longestPalindrome('abcddcba'))