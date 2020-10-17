#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def isFlipedString(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # 如果s2是由s1旋转得到，那么s1+s1中必然可以找到s2
        if len(s1) != len(s2):
            return False
        return (s1+s1).index(s2) != -1