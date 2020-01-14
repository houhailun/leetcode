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
    def getSum(self, a, b):
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

        # 循环
        sum = a
        carry = b
        while carry != 0:
            sum = sum ^ carry
            carry = (sum & carry) << 1
        return sum

cls = Solution()
print(cls.getSum(-1, 1))