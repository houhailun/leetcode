#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/20 14:53
# Author: Hou hailun

"""
对于字符串 S 和 T，只有在 S = T + ... + T（T 与自身连接 1 次或多次）时，我们才认定 “T 能除尽 S”。
返回字符串 X，要求满足 X 能除尽 str1 且 X 能除尽 str2。
示例 1：
输入：str1 = "ABCABC", str2 = "ABC"
输出："ABC"
示例 2：
输入：str1 = "ABABAB", str2 = "ABAB"
输出："AB"
示例 3：
输入：str1 = "LEET", str2 = "CODE"
输出：""
"""


class Solution(object):
    def gcdOfStrings(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: str
        """
        # 求最大公约数/最大公因子可以考虑辗转相除法
        if len(str1) < len(str2):
            str1, str2 = str2, str1  # 把短的字符串赋给str2
        len_str1 = m = len(str1)
        len_str2 = n = len(str2)
        while n > 0:
            m, n = n, m % n
        gcd = m

        # 判断最大公因子对应的前缀是否相等，以及是否能构成字符串str1和str2
        rep1, rep2 = str1[:gcd], str2[:gcd]  # 最大公因子
        if rep1 == rep2 and len_str1 // gcd * rep1 == str1 and len_str2 // gcd * rep2 == str2:
            return rep1
        return ""


obj = Solution()
print(obj.gcdOfStrings('ABABAB', 'ABAB'))