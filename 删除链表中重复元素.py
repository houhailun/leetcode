#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述：给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
示例 1:
输入: 1->1->2
输出: 1->2
示例 2:
输入: 1->1->2->3->3
输出: 1->2->3

解题思路：三个指针pre,cur,last；pre指向前一个节点，cur指向当前节点，last指向后一个节点
    1、当cur.val == pre.val：last后移，直到不等
    2、cur.next = last,pre.next = cur,last=last.next
    3、结束条件：cur.next == None
"""


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        if not head.next:
            return head

        pre, cur, last = head, head, head.next
        while cur:
            # 后移，直到不相等
            while last and cur.val == last.val:
                last = last.next
            cur.next = last
            cur = last
            if last:
                last = last.next

        return pre