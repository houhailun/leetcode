#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
# 示例:
# 输入: 1->2->3->4->5->NULL
# 输出: 5->4->3->2->1->NULL


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 方法：三指针法
        if not head:
            return head
        pPre = None
        pCur = head
        while pCur:
            pLast = pCur.next
            pCur.next = pPre
            pPre = pCur
            pCur = pLast

        return pPre