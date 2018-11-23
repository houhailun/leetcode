#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述: 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
push(x) -- 将元素 x 推入栈中。
pop() -- 删除栈顶的元素。
top() -- 获取栈顶元素。
getMin() -- 检索栈中的最小元素。
示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.

解题思路：利用辅助栈，来保存当前的栈中最小元素
    1、开始入栈val，数据栈和辅助栈都保存val
    2、再次入栈val，数据栈入栈，辅助栈入栈min(当前最小值,val)
    3、出栈，两个栈均正常出栈
"""


class MinStack:
    def __init__(self):
        self.min_stack = []   # 辅助栈
        self.data_stack = []  # 数据栈

    def push(self, x):
        self.data_stack.append(x)

        if not self.min_stack:
            self.min_stack.append(x)
        else:
            if self.min_stack[-1] > x:
                self.min_stack.append(x)
            else:
                self.min_stack.append(self.min_stack[-1])

    def pop(self):
        if self.data_stack:
            self.data_stack.pop()
            self.min_stack.pop()

    def top(self):
        if self.data_stack:
            return self.data_stack[-1]
        return None

    def getMin(self):
        if self.min_stack:
            return self.min_stack[-1]
        return None

obj = MinStack()
obj.push(3)
obj.push(2)
obj.pop()
param_3 = obj.top()
param_4 = obj.getMin()
print(param_3, param_4)