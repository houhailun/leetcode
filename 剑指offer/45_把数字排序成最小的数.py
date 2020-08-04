#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import sys
from functools import cmp_to_key

class Solution(object):
    def minNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        # 自定义比较规则
        # x+y < y+x -> x < y
        # return ''.join(sorted([str(i)for i in nums], self.cmps))
        strs = [str(num) for num in nums]
        strs.sort(key=cmp_to_key(self.sort_rule))
        return ''.join(strs)


    # 自定义排序规则
    # py2中可以使用cmp指定自定义排序函数，py3中没有了cmp参数
    # def cmps(self, x, y):
    #     return cmp((x+y), (y+x))

    # py3使用collection

    def sort_rule(self, x, y):
        if x+y < y+x:
            return -1
        elif x+y > y+x:
            return 1
        else:
            return 0

obj = Solution()
print(obj.minNumber([10, 2]))