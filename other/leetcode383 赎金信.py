#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串ransom能不能由第二个字符串magazines里面的字符构成。如果可以构成，返回 true ；否则返回 false。
(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。)
注意：
你可以假设两个字符串均只含有小写字母。
canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true
"""


class Solution:
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        # 只要第一个字符串中的字符出现次数小于在第二个字符串中的次数即可，不要求是子串
        have_done = []
        for i in range(len(ransomNote)):
            if ransomNote[i] not in have_done:
                if ransomNote.count(ransomNote[i]) <= magazine.count(ransomNote[i]):
                    have_done.append(ransomNote[i])
                else:
                    return False
        return True

cls = Solution()
print(cls.canConstruct('fffbfg', 'effjfggbffjdgbjjhhdegh'))