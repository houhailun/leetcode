#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述: 给定一个链表，判断链表中是否有环。
进阶：
你能否不使用额外空间解决此题？

解题思路：两个指针slow,fast，slow每次走一步，fast每次走两步，如果有环则必然在环内相遇

"""

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head and not head.next:
            return False
        slow = head
        fast = head.next
        # 循环终止条件：fast为空或者fast.next为空（最后一个节点）
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False
