#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 16:37
# Author: Hou hailun

# 两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目


class Solution:
    def hammingDistance(self, x, y):
        # 思路：求二进制不同的个数
        # 1、异或，不同位将变为1
        # 2、统计二进制中1的个数
        n = x ^ y
        cnt = 0
        while n:
            n = n & (n-1)
            cnt += 1
        return cnt

        # 方法2
        # return bin(x^y).count('1')

obj = Solution()
print(obj.hammingDistance(1, 4))