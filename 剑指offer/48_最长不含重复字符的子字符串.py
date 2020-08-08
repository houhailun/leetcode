#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 方法：res记录最长无重复字串，初始为空，遍历s，当前字符在res中则说明有重复，res删除第一个ch；不重复添加到res中，更新最大无重复长度
        if not s:
            return 0

        res = ''
        max_len = 0
        for ch in s:
            if ch not in res:
                res += ch
                max_len = max(len(res), max_len)
            else:
                res += ch
                res = res[res.index(ch)+1:]
        return max_len

obj = Solution()
print(obj.lengthOfLongestSubstring("abcabcbb"))