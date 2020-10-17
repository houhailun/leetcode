#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def constructArr(self, a):
        """
        :type a: List[int]
        :rtype: List[int]
        """
        # 题目的意思：b[i]等于A数组中取出A[i]后剩余元素的乘积
        # o(N*N)
        # if not a:
        #     return a
        # _len = len(a)
        # res = []
        # for i in range(_len):
        #     b = 1
        #     for j, num in enumerate(a):
        #         if j != i:
        #             b *= num
        #     res.append(b)
        # return res

        # 构建上三角和下三角矩阵
        if not a:
            return a

        _len = len(a)
        res = [1] * _len

        # 上三角
        for i in range(1, _len):
            res[i] = res[i - 1] * a[i - 1]

        # 下三角
        tmp = 1
        for i in range(_len - 2, -1, -1):
            tmp *= a[i + 1]
            res[i] = res[i] * tmp
        return res

print([1,2,3,4,5])