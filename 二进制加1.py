#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math

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
        if not a:
            return b
        if not b:
            return a
        '''
        a = int(a)
        b = int(b)
        print('a:%d, b:%d' % (a, b))
        while b:
            a = a ^ b
            b = (a & b) << 1
        return str(a)
        '''

        # 方法2：利用python内置函数实现
        # a = int(a, 2)
        # b = int(b, 2)
        # res = bin(a+b)[2:]
        # return res
        # 利用辗转相除法：二进制和十进制的转换
        a = self.convert_int(a)
        b = self.convert_int(b)
        res = a + b
        return self.int_convert(res, 2)

    def int_convert(self, n, x):
        """
        10进制转换为x进制，辗转相除法：不断求商和余数，直到商为0，余数翻转
        :param n: 待转换为十进制数字
        :param x: 想要转换的进制，可以为2，8，10，16
        :return:
        """
        lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F']
        tmp = []
        while True:
            s = n // x
            d = n % x
            tmp.append(d)
            if s == 0:
                break
            n = s
        tmp.reverse()

        res = ''
        for i in tmp:
            res += str(lst[i])
        return res

    def convert_int(self, n):
        """
        2进制转换为10进制
        :param n: 待转换为字符串
        :return:
        """
        sum_n = 0
        n = int(n)  # 输入的是二进制字符串
        i = 0
        while n:
            desc = n % 2
            sum_n += desc * math.pow(2, i)
            n = n // 10
            i += 1
        return int(sum_n)


cls = Solution()
print(cls.addBinary('1010', '1011'))

# print(cls.convert_int("1100"))