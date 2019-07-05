#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
题目描述：实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
示例 1:
输入: 4
输出: 2
示例 2:
输入: 8
输出: 2
说明: 8 的平方根是 2.82842...,
     由于返回类型是整数，小数部分将被舍去
    
方法1：利用math.sqrt()
方法2: 利用x ** 0.5
方法3：二分法,start=0, end=n
    1、mid =  (start+end) // 2
    2、mid * mid > n: end = mid-1
    3、mid * mid < n: start = mid+1
"""

import math


class Solution:
    def my_sqrt(self, n):
        # 利用math.sqrt函数
        # return int(math.sqrt(n))
        # return int(x ** 0.5)

        # 自己实现sqrt函数：利用二分法
        low, high = 0, n
        while low <= high:
            mid = (low + high) // 2
            if mid * mid == n:
                return mid
            elif mid*mid > n:
                high = mid - 1
            else:
                low = mid + 1
        return high


cls = Solution()
print(cls.my_sqrt(12))