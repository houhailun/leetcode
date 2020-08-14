#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/8/9 11:39
# Author: Hou hailun

import queue

# 本题和"包含min函数的栈"相似
class MaxQueue(object):

    def __init__(self):
        self.queue = queue.Queue()         # 单端队列
        self.deque = queue.deque()         # 双端队列


    def max_value(self):
        """
        :rtype: int
        """
        # 最大元素始终在双端队列头部
        if self.deque:
            return self.deque[0]
        return -1

    def push_back(self, value):
        """
        :type value: int
        :rtype: None
        """
        # 把双端队列中小于下一大值的删除
        while self.deque and self.deque[-1] < value:
            self.deque.pop()
        self.deque.append(value)
        self.queue.put(value)

    def pop_front(self):
        """
        :rtype: int
        """
        if not self.deque:
            return -1
        ans = self.queue.get()
        if ans == self.deque[0]:  # 删除的最大值
            self.deque.popleft()
        return ans

obj = MaxQueue()


# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()