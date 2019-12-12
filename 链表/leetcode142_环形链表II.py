#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/14 14:04
# Author: Hou hailun

"""
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
说明：不允许修改给定的链表。
 
示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。
"""

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 思路：
        #   1、首先检查是否有环: 快慢指针实现；
        #   2、检查环内的节点数N: 在环中的某节点相遇后，p1保持不变，p2循环，再次相遇后即可知道环中节点个数；
        #   3、快慢指针，快指针先走N步，然后两个指针一起走，直到相等为止
        if head is None or head.next is None:
            return -1

        node = self.hasCycle(head)
        if node is None:
            return -1

        node_nums = self.CycleNodeNums(node)

        return self.firstDetectCycle(head, node_nums)

    def hasCycle(self, head):
        # 检查是否有环, 快慢指针实现
        fast_node = head.next
        slow_node = head
        # 循环终止条件：fast为空或者fast.next为空（最后一个节点）
        while fast_node and fast_node.next:
            slow_node = slow_node.next
            fast_node = fast_node.next.next
            if fast_node == slow_node:
                return fast_node
        return None

    def CycleNodeNums(self, node):
        # 计算环内的节点个数
        slow = node
        node_nums = 1
        while slow.next != node:
            node_nums += 1
            slow = slow.next
        return node_nums

    def firstDetectCycle(self, head, node_nums):
        # 链表环的入口节点
        fast = head
        slow = head
        for i in range(node_nums):
            fast = fast.next

        i = 0
        while fast != slow:
            fast = fast.next
            slow = slow.next
            i += 1
        return i
