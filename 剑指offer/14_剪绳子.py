#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/4 15:59
# Author: Hou hailun

"""
给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。
请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18
"""

# 动态规划
# 假设f(n)为长度为n的绳子剑成若干段后的最大乘积。那么如果我们只剪一刀，该绳子就会分为两部分，假设在i上剪断，
# 我们再求出这两部分绳子的最大乘积f(i)，f(n-i)。然后不断以相同的方式进行分割，可以得到f(n)=max(f(i)*f(n-i))。
# 我们可以用递归求出来，因为这是一个自上而下的式子，但递归有许多重复的计算，效率低。
# 这题我们采用自下而上的方式，用动态规划来做。f(1)，f(2)，f(3)都很容易得到，然后我们可以推出f(4)，f(5)，然后一直往后推出至f(n)。


class Solution(object):
    def cuttingRope(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 方法1：动态规划
        # return self.cuttingRope_do(n)

        # 方法2：贪心算法n
        return self.cuttingRope_grep(n)

    def cuttingRope_do(self, n):
        # 考虑到至少要剪1刀(m > 1)，先将剪后的f(n) < n的情况作为特例排除
        if n < 2:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        max_val = 0  # 记录当前最大f(n)
        dp = [0] * (n+1)
        # 下面几个不是乘积，因为其本身长度比乘积大
        dp[0] = 0
        dp[1] = 1
        dp[2] = 2
        dp[3] = 3
        for i in range(4, n+1):
            for f in range(1, i // 2 + 1):  # 绳子只需要剪前一半的就可以，剪i和剪n-i是一回事
                pro = dp[f] * dp[i-f]
                if pro > max_val:
                    max_val = pro
                    dp[i] = max_val

        return dp[n]

    def cuttingRope_grep(self, n):
        # 当n<=3时，不再进行剪切，因为会比n小
        if n < 2:
            return 0
        if n == 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 4
        max_ = 1
        # 当n>=5时，我们要把所有的绳子都剪成2或者3，同时我们要尽可能的多剪长度为3的绳子，因为3(n-3)>=2(n-2)，当剩余的小于5时就没有必要再剪
        while n >= 5:
            n -= 3
            max_ *= 3
        if n != 0:
            max_ *= n
        return max_


obj = Solution()
print(obj.cuttingRope(2))
