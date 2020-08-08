#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return " "
        # 查找字符/数字出现的次数，统一优先考虑哈希表
        hash_table = {}
        for ch in s:
            if ch not in hash_table:
                hash_table[ch] = 1
            else:
                hash_table[ch] += 1

        print(hash_table)
        for ch in s:
            if hash_table[ch] == 1:
                return ch
        return " "

obj = Solution()
print(obj.firstUniqChar("leetcode"))
