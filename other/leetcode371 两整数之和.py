#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
不使用运算符 + 和 - ​​​​​​​，计算两整数 ​​​​​​​a 、b ​​​​​​​之和。
示例 1:
输入: a = 1, b = 2
输出: 3
示例 2:
输入: a = -2, b = 3
输出: 1
"""


class Solution(object):
    def getSum(self, num1, num2):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # a ^ b：无进位的相加
        # a & b<<1 : 表示进位
        # 递归
        # sum = a ^ b
        # carry = (a & b) << 1
        # if carry != 0:
        #     return self.getSum(sum, carry)
        # return sum

        if num1 == 0 or num2 == 0:
            return num1 or num2

        while num2:
            _sum = num1 ^ num2
            carry = (num1 & num2) << 1  # 进位左移

            num1 = _sum
            num2 = carry
        return num1

cls = Solution()
print(cls.getSum(7, 2))