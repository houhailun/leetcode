#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class MedianFinder(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.nums = []
        self.size = 0


    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        self.nums.append(num)
        self.size += 1


    def findMedian(self):
        """
        :rtype: float
        """
        if self.size == 0:
            return None
        if self.size & 0x01 == 1:  # 奇数
            return self.nums[self.size // 2]
        return (self.nums[(self.size-1) // 2] + self.nums[(self.size+1)//2]) / 2


obj = MedianFinder()
obj.addNum(1)
obj.addNum(2)
print(obj.findMedian())