#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def compressString(self, S):
        """
        :type S: str
        :rtype: str
        """
        # 思路: 指针直接移动到与当前值不同的值，并记录重复出现的次数、
        if not S:
            return ""
        res = ""
        len_s = len(S)
        i = 0
        while i < len_s:
            j = i + 1
            while j < len_s:
                if S[i] == S[j]:
                    j += 1
                else:
                    break
            res += S[i]
            res += str(j - i)
            print(res)
            i = j
        return res


obj = Solution()
print(obj.compressString("aabcccccaaa"))