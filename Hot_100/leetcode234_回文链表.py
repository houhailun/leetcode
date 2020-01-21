#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 13:59
# Author: Hou hailun

# 请判断一个链表是否是回文链表

class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 方法1：辅助列表+快慢指针
        # 遍历链表，把链表元素复制到列表中，最后判断列表是否回文
        tmp = []
        cur = head
        while cur:
            tmp.append(cur.val)
            cur = cur.next

        # start, end = 0, len(tmp) - 1
        # while start < end:
        #     if tmp[start] != tmp[end]:
        #         return False
        #     start += 1
        #     end -= 1
        # return True

        # 简单可以
        return tmp == tmp[::-1]


# 扩展
# 回文数
# 判断x是否是回文数
# s = str(x)
# return s == s[::-1]

# 回文串
# 判断字符串s是否是回文串
def isPalindrome(self, s):
    """
    :type s: str
    :rtype: bool
    """
    if not s:
        return True

    # tmp = []
    # s = s.lower()
    # pattern = 'abcdefghijklmnopqrstuvwxyz0123456789'
    #
    # for st in s:
    #     if st in pattern:
    #         tmp.append(st)
    #
    # return tmp == tmp[::-1]

    # 第2种方法：简单，速度快
    new_str = filter(str.isalnum(), s.lower())
    return new_str == new_str[::-1]