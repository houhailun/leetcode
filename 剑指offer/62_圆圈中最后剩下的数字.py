#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution:
    def lastRemaining(self, n, m):
        ix = 0  # 初始下标
        _list = list(range(n))  # 长度为n的环
        while len(_list) > 1:
            ix = (ix + m - 1) % len(_list)  # 每次走m步，走到最后要从头开始
            _list.pop(ix)  # pop()后后面的元素会移动到前面，ix不变
        return _list[0]
