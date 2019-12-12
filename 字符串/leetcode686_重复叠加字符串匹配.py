#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/20 11:23
# Author: Hou hailun

"""
给定两个字符串 A 和 B, 寻找重复叠加字符串A的最小次数，使得字符串B成为叠加后的字符串A的子串，如果不存在则返回 -1。
举个例子，A = "abcd"，B = "cdabcdab"。
答案为 3， 因为 A 重复叠加三遍后为 “abcdabcdabcd”，此时 B 是其子串；A 重复叠加两遍后为"abcdabcd"，B 并不是其子串。
注意: A 与 B 字符串的长度在1和10000区间范围内。
"""


class Solution:
    def repeatedStringMatch(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: int
        """
        # 如果A的长度比B的长度小，那么A需要加大到至少比B的长度大，所以一开始在A长度比B长度小的时候，要不断加大A，并记录下加大的次数
        # 当A的长度比B大时，先判断B是否是A的子串：
        #   如果是，则直接返回i
        #   如果不是，就把A再加大一次，再判断B是否是加大后的A的子串：
        #       如果是，返回i
        #       如果不是，则直接返回 - 1，因为在A第一次长度比B大的时候，有可能因为头尾没接上的原因使得B不是A的子串，
        #       在A又加大了一次后，解决了头尾没接上这个要素，所以如果此时B仍不是A的子串，那么无论A再怎么加，B都不可能是A的子串。
        s = A
        r = 1
        while len(s) < len(B):
            s += A
            r += 1
        if B in s:
            return r
        s += A
        if B not in s:
            return -1
        return r + 1


obj = Solution()
A = "abcd"
B = "cdabcdab"
print(obj.repeatedStringMatch(A, B))