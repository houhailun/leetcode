#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def translateNum(self, num):
        """
        :type num: int
        :rtype: int
        """
        # 动态规划问题
        # dp[0]=dp[1]=1
        # dp[i]表示从开始到第i位数字的翻译方案数量
        # dp[i] = dp[i-1]+dp[i-2]，条件:Xi-2Xi-1能整体翻译，[10,25]
        # dp[i] = dp[i-1], other
        s = str(num)
        if len(s) < 2:
            return 1
        dp = [0] * len(s)
        dp[0] = 1
        dp[1] = 2 if int(s[0] + s[1]) < 26 else 1
        for i in range(2, len(s)):
            dp[i] = dp[i - 1] + dp[i - 2] if int(s[i - 1] + s[i]) < 26 and s[i - 1] != '0' else dp[i - 1]
        return dp[-1]


obj = Solution()
print(obj.translateNum(12258))