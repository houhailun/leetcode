# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        # 快慢指针法
        # step1：确认两个链表的长度
        # step2: 长的链表先走abs(len1-len2)步
        # step3: 两个链表一起走，第一次相遇的节点即为第一个公共节点
        # if not headA or not headB:
        #     return None
        # len1 = len2 = 0
        # node = headA
        # while node:
        #     len1 += 1
        #     node = node.next
        # node = headB
        # while node:
        #     len2 += 1
        #     node = node.next

        # node1, node2 = headA, headB
        # if len1 > len2:
        #     for i in range(len1 - len2):
        #         node1 = node1.next
        # else:
        #     for i in range(len2 - len1):
        #         node2 = node2.next

        # while node1 != node2:
        #     node1 = node1.next
        #     node2 = node2.next
        # return node1

        # 方法2：走别人的路
        lA, lB = headA, headB
        while lA != lB:
            lA = lA.next if lA else headB
            lB = lB.next if lB else headA
        return lA