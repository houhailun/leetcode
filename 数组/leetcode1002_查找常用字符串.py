#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 17:06
@Author  : Hou hailun
@File    : leetcode1002_查找常用字符串.py
"""

print(__doc__)

import collections


class Solution(object):
    def commonChars(self, A):
        """
        :type A: List[str]
        :rtype: List[str]
        """
        # 利用dict的键值对，先把每个字符的重复情况找出来放在dic里，然后遍历整个字符串，用min来取交集统计重复的字符
        n = len(A)
        dic = [collections.Counter(A[i]) for i in range(n)]
        compare = dic[0]

        for i in range(1, n):
            for key in compare:
                compare[key] = min(compare[key], dic[i][key])

        res = list()
        for key in compare:
            for i in range(compare[key]):
                res.append(key)
        return res


obj = Solution()
ret = obj.commonChars(["bella","label","roller"])
print(ret)