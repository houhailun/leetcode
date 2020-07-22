#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/19 17:41
# Author: Hou hailun

class MyCircularQueue(object):

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self.arr = [None] * (k + 1)
        self.front = 0
        self.rear = 0

    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        # 队列满则返回False
        if self.isFull():
            return False

        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % len(self.arr)
        return True

    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        # 队列空返回False
        if self.isEmpty():
            return False

        self.front = (self.front + 1) % len(self.arr)
        # self.arr.pop(0)
        return True

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        # 队列空返回-1
        if self.isEmpty():
            return -1

        return self.arr[self.front]

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        # 队列空返回-1
        if self.isEmpty():
            return -1

        return self.arr[(self.rear - 1) % len(self.arr)]

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return self.front == self.rear

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        if len(self.arr) > 0:
            return (self.rear + 1) % len(self.arr) == self.front
        return False


# Your MyCircularQueue object will be instantiated and called as such:
obj = MyCircularQueue(10)
param_1 = obj.enQueue(7)
param_2 = obj.deQueue()
param_3 = obj.Front()
param_4 = obj.Rear()
param_5 = obj.isEmpty()
param_6 = obj.isFull()

print(param_1, param_2, param_3, param_4, param_5, param_6)