#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目名称：有效的括号

题目描述：给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
注意空字符串可被认为是有效字符串。
示例 1:
输入: "()"
输出: true
示例 2:
输入: "()[]{}"
输出: true
示例 3:
输入: "(]"
输出: false
示例 4:
输入: "([)]"
输出: false
示例 5:
输入: "{[]}"
输出: true

解题思路：若为对称括号，则必然左括号和右括号是都存在的，且前面的左括号对应的右括号必然在后面，也就是([])
    1、遇到左括号入栈
    2、遇到右括号出栈
    3、栈为空咋对称
"""

class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 用时：48ms
        dic = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for cur in s:
            if cur in dic.keys():  # 左括号入栈
                stack.append(cur)
            elif cur in dic.values():  # 右括号出栈
                # 分别对应如 []}, [} 这两种情况
                if len(stack) == 0 or dic.get(stack[-1]) != cur:
                    return False
                stack.pop()

        return True if len(stack) == 0 else False

        # 优秀代码
        if not s:
            return True
        if len(s) % 2 == 1:
            return False
        dic = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for each in s:
            if each in dic:
                stack.append(each)
            else:
                if not stack or dic[stack.pop()] != each:
                    return False
        return stack == []

cls = Solution()
print(cls.isValid(")"))
