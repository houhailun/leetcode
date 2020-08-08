#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def kthLargest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        # 二叉搜索树：做孩子节点值 < 跟节点值 < 右孩子节点值
        # 中序遍历可得到有序列表
        # if not root:
        #     return
        # res = []
        # def helper(root):
        #     if not root:
        #         return
        #     helper(root.left)
        #     res.append(root.val)
        #     helper(root.right)
        # helper(root)
        # return res[len(res)-k]

        # 方法2：不需要遍历完整个树，在遍历的时候标记当前是第几大节点
        def dfs(root):
            if not root: return
            dfs(root.right)
            if self.k == 0: return
            self.k -= 1
            if self.k == 0: self.res = root.val
            dfs(root.left)

        self.k = k
        dfs(root)
        return self.res