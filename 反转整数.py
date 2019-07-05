#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目：反转整数
题目描述：给定一个 32 位有符号整数，将整数中的数字进行反转。
示例 1: 输入: 123  输出: 321
示例 2: 输入: -123  输出: -321
示例 3: 输入: 120 输出: 21
注意:假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231,  231 − 1]。根据这个假设，如果反转后的整数溢出，则返回 0。
"""
import numpy as np

class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        # leetcode上通过，但是IDE编译失败：RuntimeWarning: overflow encountered in int_scalars
        '''
        if x < np.power(-2, 31) or x > np.power(2, 31)-1:
            return 0

        str_x = str(x)
        flag = False
        ret = str_x
        if str_x[0] == '-':
            flag = True
            ret = str_x[1:]
        ret = ret[::-1]
        ret = int('-'+ret) if flag else int(ret)
        if ret < np.power(-2, 31) or ret > np.power(2, 31)-1:
            return 0
        return ret
        '''

        # 利用 pop = x % 10, x /= 10
        '''
        print('x:', x)
        flag = False
        if x < 0:
            flag = True
            x = 0 - x
        res = 0
        while x:
            pop = x % 10
            x //= 10
            res = res * 10 + pop

        if flag:
            res = 0 - res

        if res < np.power(-2, 31) or res > np.power(2, 31)-1:
            return 0
        return res
        '''

        # 别人的好代码
        reverse_num = int(str(abs(x))[::-1])
        if reverse_num.bit_length() > 31:
            return 0
        return reverse_num if x > 0 else -reverse_num

cls = Solution()
a = cls.reverse(-1200)
print(a)