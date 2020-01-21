#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/19 14:28
# Author: Hou hailun

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        # 本题目要求找两个链表的公共节点
        # 方法1：利用栈的先进后出原则，缺点：额外需要空间
        # 方法2：
        #   step1：分别求的2个链表的长度len1，len2
        #   step2：长的链表先走(len1 - len2)步
        #   step3：2个链表同时走，第一个相同节点即为公共节点

        # 异常检查
        if not headA or not headB:
            return None

        lenA = lenB = 0
        nodeA, nodeB = headA, headB

        # step1
        while nodeA:
            lenA += 1
            nodeA = nodeA.next
        while nodeB:
            lenB += 1
            nodeB = nodeB.next

        # step2
        head_longer = headA if lenA > lenB else headB
        head_shorter = headB if lenA > lenB else headA
        for i in range(abs(lenA - lenB)):
            head_longer = head_longer.next

        # step3
        while head_longer != head_shorter and head_longer and head_shorter:
            head_longer = head_longer.next
            head_shorter = head_shorter.next

        if head_longer == head_shorter:
            return head_longer
        return None