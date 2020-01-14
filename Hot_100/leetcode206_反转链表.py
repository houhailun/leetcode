#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/14 16:32
# Author: Hou hailun

"""
反转一个单链表。
示例:
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？
"""


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype:
        """
        # 方法1：利用链表实现
        # 方法2：每次循环都把当前节点指向上一节点，以实现反转链表
        # if head is None or head.next is None:
        #     return head
        #
        # pre_node = None
        # cur_node = head
        # while cur_node:
        #     tmp_node = cur_node.next  # 保存当前节点的下一节点
        #     cur_node.next = pre_node  # 当前节点指向前面的节点
        #     pre_node = cur_node       # 前指针后移
        #     cur_node = tmp_node       # 当前指针后移
        # return pre_node

        # 方法3：尾递归
        return self.reverse(None, head)

    def reverse(self, pre_node, cur_node):
        if cur_node is None:
            return pre_node
        last_node = cur_node.next
        cur_node.next = pre_node
        return self.reverse(cur_node, last_node)