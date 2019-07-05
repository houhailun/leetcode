#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
给定两个字符串 s 和 t，它们只包含小写字母。
字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
请找出在 t 中被添加的字母。
示例:
输入：
s = "abcd"
t = "abcde"
输出：e
解释：'e' 是那个被添加的字母。
"""


class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # 利用异或性质：相同为0，不同为1,s和t中的字符异或，最后即为添加的字母
        # 类似于只有一个数字出现1次，其他数字出现偶数次
        ci = ord(t[-1])
        for i in range(len(s)):
            ci ^= ord(s[i])
            ci ^= ord(t[i])
        return chr(ci)
