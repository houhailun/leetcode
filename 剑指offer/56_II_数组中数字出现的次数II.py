#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 方法1：哈希表
        # 方法2：公式法
        return (3*sum(set(nums)) - sum(nums)) / 2