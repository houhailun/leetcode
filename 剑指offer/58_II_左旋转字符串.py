#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution:
    def reverseLeftWords(self, s, n):
        # 方法1: 左右字串拼接
        if not s:
            return s
        _len = len(s)
        n = n % _len
        return s[n:] + s[:n]

        # 方法2: 先翻转整个字符串，然后对前_len-n个，后n个分别翻转
        s = s[::-1]
        n = n % _len
        return s[:_len - n][::-1] + s[_len - n:][::-1]