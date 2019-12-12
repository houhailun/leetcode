#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/12 17:46
# Author: Hou hailun

"""
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
示例 1:
输入: "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
"""

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # return self.lengthOfLongestSubstring_v1(s)
        return self.lengthOfLongestSubstring_v2(s)

    def lengthOfLongestSubstring_v1(self, s):
        # 暴力法，双重循环
        i = max_len = 0
        size = len(s)
        for j in range(size):      # 第一层循环：控制起始位置
            for k in range(i, j):  # 第二层循环：控制从某一位置到j的字符，判断是否存在重复
                if s[k] == s[j]:   # 存在重复，则i为当前重复位置k的下一个位置
                    i = k + 1
                    break
            if j - i + 1 > max_len:
                max_len = j - i + 1
        return max_len

    def lengthOfLongestSubstring_v2(self, s):
        # 利用字符串
        length = 0
        tmp = ''
        for ch in s:
            if ch not in tmp:  # 当前字符非重复，则更新tmp，更新当前最大非重复子串长度
                tmp += ch
                length = max(length, len(tmp))
            else:              # 当前字符有重复，则更新tmp，只要重复元素右边的子串
                tmp += ch
                tmp = tmp[tmp.index(ch)+1:]
        return length


obj = Solution()
str_test = 'abbc'
print(obj.lengthOfLongestSubstring(str_test))