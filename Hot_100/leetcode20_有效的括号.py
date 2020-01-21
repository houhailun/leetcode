#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 13:37
# Author: Hou hailun

"""
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。
示例 1: 输入: "()"  输出: true
示例 2: 输入: "()[]{}"  输出: true
示例 3: 输入: "(]"  输出: false
示例 4: 输入: "([)]"    输出: false
"""
# 思路：利用辅助栈，先进后出，后进先出
#   当前出栈括号是否和下一个


class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 利用替换思想：把括号替换为空串，如果括号匹配，则最后必然等于空串
        return isValid_v1(s)

    def isValid_v1(self, s):
        while '{}' in s or '[]' in s or '()' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''

    def isValid_v2(self, s):
        stack = []
        mapping = {')': '(',
                   ']': '[',
                   '}': '{'}
        for char in s:
            # 当前字符是右括号
            if char in mapping:
                # 栈顶元素出栈，比较是否匹配
                top_elem = stack.pop() if stack else '#'
                if mapping[char] != top_elem:
                    return False
            else:
                stack.append(char)  # 入栈

        return not stack