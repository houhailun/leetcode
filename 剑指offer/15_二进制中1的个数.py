#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

# 解题思路：
#   1、利用1 & 1 = 0, 1 & 0 = 0的性质，左移或者右移，统计1的个数
#   2、利用n & (n-1) 后最尾部的1变为0的性质
#   3、python语言特性: bin(n).count('1')


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        cnt = 0
        while n:
            cnt += 1
            n = n & (n-1)
        return cnt


obj = Solution()
n = 5
print(obj.hammingWeight(n))
