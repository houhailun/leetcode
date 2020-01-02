#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/31 15:26
# Author: Hou hailun
"""
给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
示例：
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # 需要记录是否进位 carry = res / 10, 记录余数res % 10;
        # 遍历链表，判断是否进位；无进位则直接累计后写到新链表；
        # 有进位则把余数写到新链表位置，然后针对2个链表的后一位数据和进位相加
        new_head = ListNode(0)
        cur_node = new_head
        carry = 0
        while l1 or l2:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            res = x + y + carry

            carry = res // 10
            node = ListNode(res % 10)
            cur_node.next = node
            cur_node = cur_node.next

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        # 最高位进位
        if carry > 0:
            node = ListNode(1)
            cur_node.next = node
        return new_head.next