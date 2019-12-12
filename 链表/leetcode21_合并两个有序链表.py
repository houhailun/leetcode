#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目名称：合并两个有序链表

题目描述：将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
示例：
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

解题思路：4个指针，head指向新链表头结点，p1指向链表l1，p2指向链表l2，cur指向新链表当前结点
    1、最开始head = null，p1=l1,p2=l2,cur=null
    2、p1 < p2：head.next = p1,cur.next=p1,p1=p1->next     cur = cur.next
    3、p1 > p2:head不变.cur->next=p2,p2=p2.next
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 自己的代码：用时56ms
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = ListNode(None)
        cur = ListNode(None)
        p1, p2 = l1, l2
        if not l1 and not l2:
            return None
        if not l1:
            return l2
        if not l2:
            return l1
        flag = True
        while p1 and p2:
            if p1.val <= p2.val:
                if flag:  # head指针只移动一次
                    head.next = p1
                    head = head.next
                cur.next = p1
                p1 = p1.next
            else:
                if flag:
                    head.next = p2
                    head = head.next
                cur.next = p2
                p2 = p2.next

            cur = cur.next
            flag = False

        if p1:
            cur.next = p1
        if p2:
            cur.next = p2

        return head

    # 优化的代码
    def mergeTwoLists_v2(self, l1, l2):
        if not l1 or not l2:
            return l1 or l2
        head = tmp = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                tmp.next = l1
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next
        tmp.next = l1 or l2
        return head.next