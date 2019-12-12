#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 16:56
# Author: Hou hailun

"""
给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。
注意：
num1 和num2 的长度都小于 5100.
num1 和num2 都只包含数字 0-9.
num1 和num2 都不包含任何前导零。
你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式。
"""


class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        # 方法1: 先转换为int，相加后转化str
        # return str(int(num1) + int(num2))

        # 方法2：转换为列表
        num1_list = list(num1)
        num2_list = list(num2)
        sum1 = sum2 = 0
        for num in num1_list:
            sum1 = sum1 * 10 + int(num)
        for num in num2_list:
            sum2 = sum2 * 10 + int(num)
        return str(sum1 + sum2)

obj = Solution()
print(obj.addStrings('111', '122'))