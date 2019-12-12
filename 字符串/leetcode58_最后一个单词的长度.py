#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 16:31
# Author: Hou hailun、

"""
题目名称：最后一个单词的长度

题目描述：给定一个仅包含大小写字母和空格 ' ' 的字符串，返回其最后一个单词的长度。
如果不存在最后一个单词，请返回 0 。
说明：一个单词是指由字母组成，但不包含任何空格的字符串。
示例:
输入: "Hello World"
输出: 5

解题思路：利用split对空格进行切分，直接len(最后一个单词)
"""

class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0
        li = s.split()
        if not li:
            return 0
        return len(s.split()[-1])