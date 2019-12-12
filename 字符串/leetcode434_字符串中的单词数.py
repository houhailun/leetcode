#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 17:16
# Author: Hou hailun

"""
统计字符串中的单词个数，这里的单词指的是连续的不是空格的字符。
请注意，你可以假定字符串里不包括任何不可打印的字符。
示例:
输入: "Hello, my name is John"
输出: 5
"""


class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 思路: 单词以空格划分，因次可以用split()切分
        # return len(s.split())

        # 方法2：判断当前是空格，下一个不是空格
        word_count = 0
        for i in range(len(s)):
            if (i == 0 or s[i-1] == ' ') and s[i] != ' ':  # 前一个是空格，后一个不是空格，表示出现单词
                word_count += 1
        return word_count