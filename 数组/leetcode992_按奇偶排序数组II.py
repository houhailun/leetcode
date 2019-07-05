#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Time    : 2019/7/2 16:28
@Author  : Hou hailun
"""

print(__doc__)

"""
给定一个非负整数数组 A， A 中一半整数是奇数，一半整数是偶数。
对数组进行排序，以便当 A[i] 为奇数时，i 也是奇数；当 A[i] 为偶数时， i 也是偶数。
你可以返回任何满足上述条件的数组作为答案。
示例：
输入：[4,2,5,7]
输出：[4,5,2,7]
解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
"""


class Solution(object):
    def sortArrayByParityII(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        # 思路:区分奇偶数，根据指定规则依次取数
        odd = [i for i in A if i % 2]
        even = [i for i in A if not i % 2]
        return [i for n in zip(even, odd) for i in n]


obj = Solution()
ret = obj.sortArrayByParityII([4,2,5,7])
print(ret)