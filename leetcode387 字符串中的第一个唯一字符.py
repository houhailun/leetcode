#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。
案例:
s = "leetcode"
返回 0.
s = "loveleetcode",
返回 2.
"""

class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 很明显，对于求字串或数字的次数类完全可以用字典来实现
        num_dict = {}
        for ch in s:
            if ch not in num_dict:
                num_dict[ch] = 1
            else:
                num_dict[ch] += 1

        for i, ch in enumerate(s):
            if num_dict[ch] == 1:
                return i
        return -1

        # 方法2：利用python的count函数
        for i, ch in enumerate(s):
            if s.count(ch) == 1:
                return i
        return -1

cls = Solution()
print(cls.firstUniqChar('leetcode'))
