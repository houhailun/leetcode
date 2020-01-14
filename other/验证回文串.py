#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
题目描述：
给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
说明：本题中，我们将空字符串定义为有效的回文串。
示例 1:
输入: "A man, a plan, a canal: Panama"
输出: true
示例 2:
输入: "race a car"
输出: false

解题思路：
    方法1：利用python的内置函数:filter,join,lower等函数实现  执行用时156ms
    方法2：利用头尾指针实现，遇到非字母和数字跳过，不考虑大小写  执行用时128ms
    方法3：先提取出字母和数字字符，然后转换为小写，判断是否回文串
"""


class Solution:
    def isPalindrome(self, str):
        if not str:
            return True

        # 利用filter函数只过滤字母和数字字符
        st = list(filter(lambda st: st if st.isdigit() or st.isalpha() else '', str))
        st = ''.join(st)      # 转换为字符串
        st = str.lower()     #  转换为小写，忽略大小写
        if st == st[::-1]:  # 是否回文
            return True
        return False

    def is_palindrom(self, s):
        if not s:
            return True

        s = s.lower()
        length = len(s)
        start, end = 0, length-1
        while start < end:
            if not s[start].isdigit() and not s[start].isalpha():
                start += 1
            elif not s[end].isdigit() and not s[end].isalpha():
                end -= 1
            elif s[start] != s[end]:
                return False
            else:
                start += 1
                end -= 1
        return True

    def is_palindrom(self, s):
        if not s:
            return True

        l = []
        s = s.lower()
        pattern = 'abcdefghijklmnopqrstuvwxyz0123456789'
        for st in s:
            if st in pattern:
                l.append(st)
        if l == l[::-1]:
            return True
        return False


cls = Solution()
print(cls.is_palindrom("A man, a plan, a canal: Panama"))