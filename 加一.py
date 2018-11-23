#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目名称：加一

题目描述：给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

示例 1:
输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。
示例 2:
输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。

解题思路：先转换为数字，然后加1，转换为列表
"""


class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        if not digits:
            return None
        res = 0
        for i in digits:
            res = res * 10 + i
        res += 1
        ret = []
        while res:
            ret.append(res % 10)
            res //= 10
        ret.reverse()
        return ret

    def plusOne_v2(self, digits):
        # 参照别人的代码进行优化
        # 优化1：由于v1是转换为数字后，在低位在前高位在后append到列表，又reverse，这些都是比较花费时间的  56ms
        plus = 1
        ret = []
        for d in digits[::-1]:  # 翻转列表，从最低位开始处理
            v = d + plus
            if v > 9:  # 表示又进位
                plus = 1
                v = v % 10
            else:
                plus = 0
            ret.insert(0, v)  # 当有进位时，先插入v
        if plus == 1:  # 最后又进位
            ret.insert(0, plus)
        return ret

cls = Solution()
print(cls.plusOne_v2([1,2,3]))