#!/usr/bin/env python
# -*- coding:utf-8 -*-

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
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        # 方法1：利用python字符串内置函数find()实现
        # return haystack.find(needle)
        '''
        # 方法2：依次对比两个字符串
        if not needle:
            return 0
        if not haystack:
            return -1
        len_hay = len(haystack)
        len_needle = len(needle)
        i = j = 0
        while i < len_hay and j < len_needle:
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                i = i - j + 1  # i 跳转到当前匹配第一个位置的下一个位置
                j = 0  # j 跳转到首字母

        while j == len_needle:
            return i-j
        return -1
        '''
        # 方法3：KMP匹配算法
        # 方法4：利用list的切分
        if not needle:
            return 0
        if not haystack:
            return -1
        len_hay = len(haystack)
        len_needle = len(needle)
        for x in range(len_needle, len_hay+1):
            if haystack[x-len_needle:x] == needle:
                return x-len_needle
        return -1


cls = Solution()
print(cls.strStr("mississippi", "issip"))