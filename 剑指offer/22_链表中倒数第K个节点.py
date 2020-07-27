#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def getKthFromEnd(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        # 方法1：两次遍历
        # 第一次遍历，计算链表的长度n
        # 第二次遍历，走n-k步
        self.getKthFromEnd_v1(head, k)

        # 方法2：双指针
        # p1先走k步，然后p1，p2一起走，p1走到最后时，p2指向倒数第k和节点

    def getKthFromEnd_v1(self, head, k):
        node = head
        n = 0
        while node:
            node = node.next
            n += 1

        if n < k:
            return None

        node = head
        for i in range(n-k):
            node = node.next
        return node