#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 11:37
# Author: Hou hailun

"""
给定一个链表，判断链表中是否有环。
为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
"""
# 思路：归属于快慢指针问题，快指针每次走2步，慢指针每次走1步，如果有环，则必然相遇

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return False

        pSlow = head
        pFast = head.next
        # 循环终止条件: pFast指针为空或者pFast.next为空
        while pFast and pFast.next:
            pFast = pFast.next.next
            pSlow = pSlow.next
            if pFast == pSlow:
                return True
        return False