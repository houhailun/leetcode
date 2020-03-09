#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/15 11:45
# Author: Hou hailun

"""
给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。
示例：
给定一个链表: 1->2->3->4->5, 和 n = 2.
当删除了倒数第二个节点后，链表变为 1->2->3->5.
"""

# 思路：正常来说删除链表是从头到尾遍历到执行位置后删除；这里要求删除倒数第N个链表
# 方法1：首先统计链表长度K；然后移动K-n步，删除
# 方法2：快慢指针，快指针先走n步，然后一起走，这样当快指针走到末尾时慢指针在倒数第n个节点上


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        # 方法1
        self.removeNthFromEnd_solution1(head, n)

        # 方法2
        self.removeNthFromEnd_solution2(head, n)

    def removeNthFromEnd_solution1(self, head, n):
        # 统计链表长度
        length = 0
        cur = head
        while cur:
            length += 1
            cur = cur.next

        # 移动n-2步
        if n > length:
            return None
        cur = head
        for i in range(n):
            cur = cur.next
        return cur

    def removeNthFromEnd_solution2(self, head, n):
        if head is None or head.next is None:
            return None
        slow = head
        fast = head
        for i in range(n):
            if fast:
                fast = fast.next

        # 如果链表长度等于n，则表示删除第一个节点
        if fast is None:
            return head.next

        while fast.next:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        return head

