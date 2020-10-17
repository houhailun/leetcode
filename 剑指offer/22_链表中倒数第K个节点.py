# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getKthFromEnd(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        # 方法1：两次遍历
        # 第一次遍历，计算链表的长度n
        # 第二次遍历，走n-k步
        # 用时20ms
        # return self.getKthFromEnd_v1(head, k)

        # 方法2：双指针
        # p1先走k步，然后p1，p2一起走，p1走到最后时，p2指向倒数第k和节点
        return self.getKthFromEnd_v2(head, k)

    def getKthFromEnd_v1(self, head, k):
        node = head
        n = 0
        while node:
            node = node.next
            n += 1

        # k不能大于链表长度
        if n < k:
            return None

        node = head
        for i in range(n - k):
            node = node.next
        return node

    def getKthFromEnd_v2(self, head, k):
        p1 = p2 = head
        for i in range(k):
            if p1:
                p1 = p1.next
        while p1:
            p1 = p1.next
            p2 = p2.next
        return p2