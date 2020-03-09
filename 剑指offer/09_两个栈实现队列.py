#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 15:27
# Author: Hou hailun

# 用两个栈实现一个队列。队列的声明如下，
# 请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。
# (若队列中没有元素，deleteHead 操作返回 -1 )
# 示例 1：
# 输入：
# ["CQueue","appendTail","deleteHead","deleteHead"]
# [[],[3],[],[]]
# 输出：[null,null,3,-1]
# 示例 2：
# 输入：
# ["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
# [[],[],[5],[2],[],[]]
# 输出：[null,-1,null,null,5,2]


class CQueue(object):
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def appendTail(self, value):
        """
        入队列操作：入栈1
        :type value: int
        :rtype: None
        """
        self.stack1.append(value)

    def deleteHead(self):
        """
        出队列：先入先出
        思路：若栈2有数据，则出栈2；否则，把栈1的数据拷贝到栈2
        :rtype: int
        """
        if self.stack2:
            return self.stack2.pop()

        while self.stack1:
            self.stack2.append(self.stack1.pop())
        if self.stack2:
            return self.stack2.pop()
        return -1


# Your CQueue object will be instantiated and called as such:
obj = CQueue()
obj.appendTail(1)
obj.appendTail(2)
obj.appendTail(3)
param_2 = obj.deleteHead()
print(param_2)