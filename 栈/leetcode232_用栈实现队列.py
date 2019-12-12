#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/18 17:07
# Author: Hou hailun

# 栈：后进先出
# 队列：先进先出
# 思路：
#   入队列：压入栈1
#   出队列：把栈1中的元素全部压入栈2，然后弹出栈2元素
# 注意：出队列时要判断栈2是否还有元素，如果有则不需要入栈2


class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.inStack = []
        self.outStack = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.inStack.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if len(self.outStack) == 0:
            while self.inStack:
                self.outStack.append(self.inStack.pop())
        return self.outStack.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if len(self.outStack) == 0:
            while self.inStack:
                self.outStack.append(self.inStack.pop())
        return self.outStack[-1]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return self.inStack == [] and self.outStack == []

# Your MyQueue object will be instantiated and called as such:
obj = MyQueue()
obj.push(1)
param_2 = obj.pop()
param_3 = obj.peek()
param_4 = obj.empty()
