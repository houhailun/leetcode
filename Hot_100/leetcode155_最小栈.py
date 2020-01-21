#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/19 15:01
# Author: Hou hailun

"""
设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

push(x) -- 将元素 x 推入栈中。
pop() -- 删除栈顶的元素。
top() -- 获取栈顶元素。
getMin() -- 检索栈中的最小元素。
"""


class MinStack(object):
    # 思路：需要辅助栈来实现当前栈内最小元素
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []         # 数据栈
        self.stack_helper = []  # 辅助栈
        self.size = 0

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.stack.append(x)

        # 若待插入值小于当前栈中最小值，则插入待插入值，否则插入栈中的最小值
        if self.size == 0:
            self.stack_helper.append(x)
        elif x < self.stack_helper[self.size-1]:
            self.stack_helper.append(x)
        else:
            self.stack_helper.append(self.stack_helper[self.size-1])
        self.size += 1

    def pop(self):
        """
        :rtype: None
        """
        if self.size != 0:
            self.stack.pop()
            self.stack_helper.pop()
            self.size -= 1

    def top(self):
        """
        :rtype: int
        """
        if self.size != 0:
            return self.stack[self.size-1]

    def getMin(self):
        """
        :rtype: int
        """
        if self.size != 0:
            return self.stack_helper[self.size-1]

# Your MinStack object will be instantiated and called as such:
obj = MinStack()
obj.push(3)
obj.push(4)
obj.push(1)
obj.push(5)
# obj.pop()
param_3 = obj.top()
param_4 = obj.getMin()

print(param_3)
print(param_4)