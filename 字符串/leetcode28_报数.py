#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 16:22
# Author: Hou hailun

"""
题目名称：报数

题目描述：报数序列是一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：
1.     1
2.     11
3.     21
4.     1211
5.     111221
1 被读作  "one 1"  ("一个一") , 即 11。
11 被读作 "two 1s" ("两个一"）, 即 21。
21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。
给定一个正整数 n（1 ≤ n ≤ 30），输出报数序列的第 n 项。
注意：整数顺序将表示为一个字符串。

示例 1:
输入: 1
输出: "1"
示例 2:
输入: 4
输出: "1211"

解题思路：本题是根据前一项来推的下一项，比如一开始输入为1，得到1->(1个1)->第二项结果为11->(2个1)->第三项结果为21->(1个2，1个1)->第四项结果为1211->(一个1，一个2，2个1)->111221->(3个1，2个2，1个1)->
    第5项结果为312211
    1、根据推导规则，可以发现需要一个times变量来记录当前st的个数，st表示当前字符
    2、检查st的times，res = str(times) + st，最后当作字符串处理
"""


class Solution:
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        # 52ms, 在Count and Say的Python3提交中击败了74.81% 的用户
        if n <= 0:
            return None
        if n == 1:
            return '1'
        if n == 2:
            return '11'
        pre = '11'  # 表示上一项
        for i in range(3, n+1):
            res = ''
            cnt = 1  # 记录次数
            length = len(pre)
            for j in range(1, length):
                if pre[j-1] == pre[j]:  # 相等则次数加1
                    cnt += 1
                else:
                    res += str(cnt)+pre[j-1]
                    cnt = 1  # 重置为1，记录下一个字符的次数
            # 最后一项
            res += str(cnt)+pre[j]
            pre = res  # 保存上一项的结果
        return res