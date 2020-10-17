#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

class Solution(object):
    def add(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # 二进制运算
        # 不考虑进位: 0+1=1，1+1=0，正好和异或相同，因此使用异或来计算
        # 考虑进位: 0+0=0,0+1=0,1+1=1 有进位，和与运算相同，进位要左移1
        # 终止条件: 没有进位为止
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a ^ x)

