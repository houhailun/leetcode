#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 18:15
# Author: Hou hailun

"""
给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成。给定的字符串只含有小写英文字母，并且长度不超过10000。
示例 1:
输入: "abab"
输出: True
解释: 可由子字符串 "ab" 重复两次构成。
示例 2:
输入: "aba"
输出: False
示例 3:
输入: "abcabcabcabc"
输出: True
解释: 可由子字符串 "abc" 重复四次构成。 (或者子字符串 "abcabc" 重复两次构成。)
"""
# 解题思路: 1、计算第一个字符的次数cnt,判断cnt*len(字符) == len(字符串)


class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 问题：当s过长时，容易导致超时
        # if not s:
        #     return False
        # for i in range(1, len(s)):
        #     ch = s[:i]
        #     count = s.count(ch)
        #     if count * len(ch) == len(s):
        #         return True
        # return False

        # 方法2：假设母串S是由子串s重复N次而成， 则 S+S则有子串s重复2N次， 现在S=Ns， S+S=2Ns 因此S在(S+S)[1:-1]中必出现一次以上
        return s in (s+s)[1:-1]

obj = Solution()
print(obj.repeatedSubstringPattern('aba'))