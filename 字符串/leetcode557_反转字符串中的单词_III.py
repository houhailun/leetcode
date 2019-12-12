#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 20:25
# Author: Hou hailun

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        # ret = ''
        # word_list = s.split()
        # for word in word_list:
        #     word = word[::-1]
        #     ret += word
        #     ret += ' '
        # return ret[:-1]

        # 优化
        word_list = s.split()
        return ' '.join(word[::-1] for word in word_list)


obj = Solution()
print(obj.reverseWords("Let's take LeetCode contest"))