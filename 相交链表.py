#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述：编写一个程序，找到两个单链表相交的起始节点。
例如，下面的两个链表：
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗
B:     b1 → b2 → b3
在节点 c1 开始相交。

注意：
如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。

 解题思路：起始就是找两个链表的第一个公共节点，
    方法1: 设置slow，fast两个指针
        1、找两个链表的长度len1，len2
        2、fast指针先走abs(len1-len2)步，然后两个指针一起走
        3、相遇的地方就是第一个公共节点
    方法2：利用set(),把A链表添加到set中，遍历B链表，找出第一个相等节点
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def get_intersection_node(self, headA, headB):
        if not headA or not headB:
            return []
        lenA = lenB = 0
        nodeA = headA
        nodeB = headB

        # 两个链表的长度
        while nodeA:
            lenA += 1
            nodeA = nodeA.next
        while nodeB:
            lenB += 1
            nodeB = nodeB.next

        # fast先走abs(lenA-lenB)
        fast = headA if lenA>lenB else headB
        slow = headB if lenA>lenB else headA
        for i in range(abs(lenA-lenB)):
            fast = fast.next

        # slow，fast一起走
        while fast != slow and fast and slow:
            fast = fast.next
            slow = slow.next

        if fast == slow:
            return fast
        return []