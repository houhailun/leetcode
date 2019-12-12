#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/22 14:29
# Author: Hou hailun

# 用两个队列实现栈


class Solution:
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def push(self, item):
        self.queue1.append(item)

    def pop(self):
        # 弹出时，把队列1中元素取出到只剩一个为止，放到队里2中
        # 队列1和2交换为止
        # 弹出队列2中的元素
        if len(self.queue1) == 0:
            return None

        while len(self.queue1) != 1:
            self.queue2.append(self.queue1.pop(0))

        self.queue1, self.queue2 = self.queue2, self.queue1
        return self.queue2.pop()

obj = Solution()
obj.push(1)
obj.push(2)
obj.push(3)
print(obj.pop())
print(obj.pop())
