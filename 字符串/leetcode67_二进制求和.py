#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 16:44
# Author: Hou hailun

"""
题目名称：二进制加1

题目描述：给定两个二进制字符串，返回他们的和（用二进制表示）。
输入为非空字符串且只包含数字 1 和 0。
示例 1:
输入: a = "11", b = "1"
输出: "100"
示例 2:
输入: a = "1010", b = "1011"
输出: "10101"

解题思路：
    方法1：类似题目(整数求和),step1:两个数异或 step2:两个数位与求进位，并左移一位  step3:重复，直到进位位0
    方法2：先转换为10进制，求和后，在转换为二进制
"""


class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a = int(a, 2)
        b = int(b, 2)
        res = bin(a+b)[2:]
        return res


obj = Solution()
print(obj.addBinary('11', '1'))