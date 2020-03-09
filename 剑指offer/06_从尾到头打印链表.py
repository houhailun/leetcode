#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 14:56
# Author: Hou hailun

"""
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
示例 1：
输入：head = [1,3,2]
输出：[2,3,1]
"""
# 思路：正常情况来说，链表都是从头到尾遍历的，本题要求从尾到头遍历
# 方法1：利用栈的后进先出特性，从头遍历链表，依次放入栈中，遍历到链表尾部后，依次出栈
# 方法2：遍历链表的过程中反转指针
# 方法3：递归法


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution(object):
    def reversePrint(self, head):
        """
        :type head: ListNode
        :rtype: List[int]
        """
        if head is None:
            return []
        if head.next is None:
            return [head.val]

        # 方法1 O(N)  O(N)
        # return self.reversePrint_helper1(head)

        # 递归法: 每次把后面的节点放到链表头部
        # O(N)
        self.reversePrint(head.next) + [head.val]

    def reversePrint_helper1(self, head):
        stack = []
        node = head
        while node:
            stack.append(node.val)
            node = node.next

        stack = stack[::-1]
        return stack


