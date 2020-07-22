#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/10 10:05
# Author: Hou hailun


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        # 思路：tmp来记录已经遍历的非重复子串，max_len来记录最大子串的长度
        # 遍历字符串，依次判断字符ch是否已经重复(即是否在tmp中查找到)
        # 非重复：ch添加到tmp中，并更新max_len
        # 重复：ch添加到tmp中，并把第一个ch剔除

        res = ""
        max_len = 0
        for ch in s:
            if ch not in res:
                res += ch
                max_len = max(max_len, len(res))
            else:
                res += ch
                res = res[res.index(ch) + 1:]
        return max_len


obj = Solution()
print(obj.lengthOfLongestSubstring("abcabcbb"))