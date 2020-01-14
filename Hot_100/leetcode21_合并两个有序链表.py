#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 15:01
# Author: Hou hailun

"""
将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
示例：
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
"""


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def mergeTwoLists(self, l1, l2):
        # 思路：4个指针，new_head指向新链表的头指针，tmp 指向当前节点，p1指向链表1，p2指向链表2
        if not l1 or not l2:
            return l1 or l2

        new_head = tmp = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                tmp.next = l1
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next

        tmp.next = l1 or l2
        return new_head.next