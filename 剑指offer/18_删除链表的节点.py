#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。
# 返回删除后的链表的头节点

# 解题思路：
#   找到待删除节点的前面节点，删除；注意删除的是否是第一个节点

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def deleteNode(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        # 异常检查
        if head is None:
            return head

        # 头节点
        if head.val == val:
            return head.next

        node = head.next
        pre_node = head
        while node:
            if node.val != val:
                pre_node, node = node, node.next
            else:
                # 删除node节点
                pre_node.next = node.next
                break
        return head
