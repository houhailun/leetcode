#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述：假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。

示例 1：
输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 阶 + 1 阶
2.  2 阶
示例 2：
输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶

解题思路：典型的斐波那契数列问题Fib(1) = 1,Fib(0)=0,Fib(2)=2,Fib(3)=3
"""


class Solution:
    def climbStairs(self, n):
        if n <= 0:
            return 0
        a, b = 0, 1
        for i in range(n):
            a, b = b, a+b
        return b

        """
        lis = []
        for i in range(n):
            if i == 0 or i == 1:  # 第1,2项 都为1
                lis.append(1)
            else:
                lis.append(lis[i - 2] + lis[i - 1])  # 从第3项开始每项值为前两项值之和
        print(lis)
        """


cls = Solution()
print(cls.climbStairs(4))