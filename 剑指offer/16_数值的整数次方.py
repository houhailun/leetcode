#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。

# 解题思路：
#   1、python语言特性：x ** n
#   2、for循环

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        # 方法1
        # return x ** n

        # 方法2：
        # 问题：当n很大时，内存问题
        # is_positive = 1  # 标记n是否是正数
        # if n < 0:
        #     is_positive = 0
        #
        # res = 1.0
        # for i in range(abs(n)):
        #     res *= x
        #
        # return res if is_positive == 1 else 1 / res

        if x == 0 and x == 0:
            return 0.0

        result = self.myPowCore(x, abs(n))
        return result if n > 0 else 1.0 / result

    def myPowCore(self, x, n):
        if n == 0:
            return 1
        if n == 1:
            return x

        result = self.myPowCore(x, n>>1)
        result *= result
        if n & 0x1 == 1:  # 最后n为1
            result *= x

        return result


obj = Solution()
x=2
n=4

print(obj.myPow(x, n))