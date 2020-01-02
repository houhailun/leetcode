#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/28 16:44
# Author: Hou hailun

"""
给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
你可以认为 s 和 t 中仅包含英文小写字母。字符串 t 可能会很长（长度 ~= 500,000），而 s 是个短字符串（长度 <=100）。
字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
"""


class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        print(self.isSubsequence_sequence(s, t))
        print(self.isSubsequence_py(s, t))

    def isSubsequence_sequence(self, s, t):
        # 顺序查找，双指针
        point1 = 0
        point2 = 0
        while point1 < len(s) and point2 < len(t):
            if s[point1] != t[point2]:
                point2 += 1
            else:
                point1 += 1
                point2 += 1
        if point1 == len(s):
            return True
        return False

    def isSubsequence_py(self, s, t):
        t = iter(t)
        return all(i in t for i in s)

    def isSubsequence_v3(self, s, t):
        # 方法二：调用str.index()
        t_index = 0
        for i in range(len(s)):
            try:
                t_index = t.index(s[i], t_index) + 1
            except ValueError:
                return False

        return True


obj = Solution()
a = 'abc'
b = 'adesbssc'
print(obj.isSubsequence(a, b))
