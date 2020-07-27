#!/usr/bin/env python
# -*- encoding:utf-8 -*-


# 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
# 输入：1->2->4, 1->3->4
# 输出：1->1->2->3->4->4


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 方法：类似于合并两个排序的数组，只不过这里不用额外空间，改变next指针即可
        p1, p2 = l1, l2
        new_head = ListNode(None)
        cur = new_head
        while p1 and p2:
            if p1.val <= p2.val:
                cur.next = p1
                p1 = p1.next
            else:
                cur.next = p2
                p2 = p2.next
            cur = cur.next

        if p1:
            cur.next = p1
        if p2:
            cur.next = p2
        return new_head.next