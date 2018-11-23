#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述：给定一个正整数，返回它在 Excel 表中相对应的列名称。
例如，
    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB
    ...
示例 1:
输入: 1
输出: "A"
示例 2:
输入: 28
输出: "AB"
示例 3:
输入: 701
输出: "ZY"

解题思路：A~Z分别为0~26，AA=26*1+1=27，类似于26进制
"""


class Solution:
    def convertToTitle(self, n):
        if n <= 0:
            return None

        num_dict = {0:'',1:'A', 2:'B',3:'C',4:'D', 5:'E', 6:'F', 7:'G',
                    8:'H', 9:'I',10:'J',11:'k',12:'L',13:'M',14:'N',
                    15:'O',16:'P',17:'Q',18:'R',19:'S',20:'T',
                    21:'U',22:'V',23:'W',24:'X',25:'Y',26:'Z'}

        res = ''
        while n:
            if n <= 26:
                res += num_dict[n]
                n = 0

            remainder = n % 26
            if remainder == 0:
                remainder = 26
                n -= 26
            res += num_dict[remainder]
            n = n // 26
        return res[::-1]


cls = Solution()
print(cls.convertToTitle(701))
