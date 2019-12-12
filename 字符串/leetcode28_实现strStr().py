#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 16:07
# Author: Hou hailun

"""
题目名称：实现strStr

题目描述：给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
示例 1:
输入: haystack = "hello", needle = "ll"
输出: 2
示例 2:
输入: haystack = "aaaaa", needle = "bba"
输出: -1
"""


class Solution:
    def strStr_1(self, haystack, needle):
        return haystack.find(needle)

    def strStr_2(self, haystack, needle):
        if not needle:
            return 0
        if not haystack:
            return -1
        len_hay = len(haystack)
        len_needle = len(needle)
        i = j = 0
        while j < len_needle and i < len_hay:
            if haystack[i] == needle[j]:  # 当前位置匹配上
                i += 1
                j += 1
            else:  # 匹配失败
                i = i - j + 1  # i跳转到当前匹配第一个位置的下一个位置
                j = 0

        if j == len_needle:  # 匹配完全
            return i - j  # 但会当前匹配的第一个位置
        return -1

    def strStr_3(self, haystack, needle):
        # 利用切片再haystack中取needle长度和needle比较
        if not needle:
            return 0
        if not haystack:
            return -1
        len_haystack, len_needle = len(haystack), len(needle)
        for x in range(len_needle, len_haystack+1):
            if haystack[x - len_needle: x] == needle:
                return x - len_needle
        return -1

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        print(self.strStr_1(haystack, needle))
        print(self.strStr_2(haystack, needle))
        print(self.strStr_3(haystack, needle))


obj = Solution()
obj.strStr(haystack = "hello", needle = "ll")
obj.strStr(haystack = "aaaaa", needle = "bba")