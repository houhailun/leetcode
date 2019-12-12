#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 20:01
# Author: Hou hailun

"""
给定一个字符串和一个整数 k，
你需要对从字符串开头算起的每个 2k 个字符的前k个字符进行反转。
如果剩余少于 k 个字符，则将剩余的所有全部反转。
如果有小于 2k 但大于或等于 k 个字符，则反转前 k 个字符，并将剩余的字符保持原样。
示例:
输入: s = "abcdefg", k = 2
输出: "bacdfeg"
"""

class Solution(object):
    def reverseStr(self, s, k):
        # 思路: 每次取2*k个元素，对前k个反转
        start, mid, end = 0, k, 2*k
        ret = ''
        while len(ret) < len(s):
            ret += s[start: mid][::-1] + s[mid: end]
            start, mid, end = start+2*k, mid+2*k, end+2*k
        return ret


obj = Solution()
print(obj.reverseStr('abcdefg', 2))