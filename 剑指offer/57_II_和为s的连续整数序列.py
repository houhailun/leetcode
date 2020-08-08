#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def findContinuousSequence(self, target):
        """
        :type target: int
        :rtype: List[List[int]]
        """
        # 双指针法，start=1，end=2
        # 1. start,...end之间元素和小于target，则end+=1
        # 2. start,...end之间元素和大于target，则start+=1
        # 3. start,...end之间元素和等于target，则记录元素，并且start+1，end+1
        if target <= 2:
            return []
        res = []
        start = 1
        end = 2
        while start <= target // 2:
            _sum = self.getSum(start, end)
            if _sum < target:
                end += 1
            elif _sum > target:
                start += 1
            else:
                res.append([i for i in range(start, end+1)])
                start += 1
                end += 1
        return res

    def getSum(self, start, end):
        res = 0
        for i in range(start, end+1):
            res += i
        return res