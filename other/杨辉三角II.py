#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
模块描述：给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。
在杨辉三角中，每个数是它左上方和右上方的数的和。
示例:
输入: 3
输出: [1,3,3,1]
要求：时间复杂度为O(k)

解题思路：
"""


class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows < 0:
            return []
        elif numRows == 0:
            return [1]
        elif numRows == 1:
            return [1, 1]

        res = [1, 1]
        numRows -= 1
        while numRows > 0:
            tmp = [1]
            for i in range(len(res)-1):
                tmp.append(res[i] + res[i+1])
            tmp.append(1)
            res = tmp
            numRows -= 1
        return res


cls = Solution()
print(cls.generate(3))