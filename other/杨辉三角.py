#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
模块描述：给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。在杨辉三角中，每个数是它左上方和右上方的数的和。
示例:
输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

解题思路：杨辉三角：两个边为1，其他数时左上方和右上方的数和
    方法1：利用杨辉三角特性：第i行，a[i][0]=a[i][i]=0; a[i][j] = a[i-1][j-1] + a[i-1][j]
    方法2：
"""


class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows <= 0:
            return []
        elif numRows == 1:
            return [[1]]
        elif numRows == 2:
            return [[1], [1, 1]]

        res = [[1], [1, 1]]
        numRows -= 2
        while numRows > 0:
            tmp = [1]
            for i in range(len(res[-1])-1):
                tmp.append(res[-1][i] + res[-1][i+1])
            tmp.append(1)
            res.append(tmp)
            numRows -= 1
        return res


cls = Solution()
print(cls.generate(5))