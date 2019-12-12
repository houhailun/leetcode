#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/14 17:35
# Author: Hou hailun

"""
请判断一个链表是否为回文链表。
示例 1:
输入: 1->2
输出: false
示例 2:
输入: 1->2->2->1
输出: true
进阶：
你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？
"""
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 方法1：辅助列表+快慢指针
        # 时间复杂度O(n), 空间复杂度O(1)
        # tmp = []
        # cur = head
        # while cur:
        #     tmp.append(cur.val)
        #     cur = cur.next
        #
        # 或者 tmp == tmp[::-1]

        # start, end = 0, len(tmp)-1
        # while start < end:
        #     if tmp[start] != tmp[end]:
        #         return False
        #     start += 1
        #     end -= 1
        # return True

        # 方法2：快慢指针+翻转
        # 时间复杂度O(n), 空间复杂度O(1)
        fast = slow = head
        # 快慢指针，快指针到达尾部，满指针到达中间
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # 奇数长，fast指针在最后一个，slow在中间，slow需要往后移动一位
        # 偶数长，fast为空，slow指针在中间后面的一位
        if fast:
            slow = slow.next
        pre = None
        cur = slow
        while cur:  # 翻转前半部分链表
            tmp = cur.next
            cur.next = pre
            pre, cur = cur, tmp

        while pre and head:
            if pre.val != head.val:
                return False
            pre = pre.next
            head = head.next
        return True

