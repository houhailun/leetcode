#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/14 16:18
# Author: Hou hailun

"""
删除链表中等于给定值 val 的所有节点。
示例:
输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5
"""

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        # 双指针问题，cur指向当前节点，pre指向上一节点
        if not head:
            return head

        first_node = ListNode(0)
        first_node.next = head
        pre_node, cur_node = first_node, head
        while cur_node:
            if cur_node.val == val:
                pre_node.next = cur_node.next
                cur_node = cur_node.next
            else:
                pre_node = pre_node.next
                cur_node = cur_node.next
        return first_node.next

        # 方法2：遇到val就后移
        # if head == None:
        #     return None
        # while head is not None and head.val == val:
        #     head = head.next
        #
        # if head == None:
        #     return None
        #
        # p = head
