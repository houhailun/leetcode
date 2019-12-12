#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/15 11:14
# Author: Hou hailun

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 快慢指针，快指针每次走2步，慢指针每次走1步
        if head is None and head.next is None:
            return head

        slow = head
        fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow