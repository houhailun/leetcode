#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        # return self.lowestCommonAncestor_helper(root, p, q)
        return self.lowestCommonAncestor_helper2(root, p, q)

    def lowestCommonAncestor_helper2(self, root, p, q):
        # 后续遍历
        # 思路：找就近的祖先节点，采用递归方法，考虑3种边界情况，
        #     边界情况：
        #       1. p,q在左右两边
        #       2. p,q都在左子树或右子树
        #       3. p或q有一个是根节点
        #       递归结束：如果not root， 找到节点为p或q，return root
        #       当前层递归：
        #       返回值：在左子树和右子树进行递归，并判断如果左子树返回值为空，则说明左子树不存在，则返回右子树，同理右子树； 如果左右子树同时存在值则返回 root为最近祖先节点
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left:
            return right
        if not right:
            return left
        return root  # 如果left 和 right都不为空，则返回根节点

    def lowestCommonAncestor_helper(self, root, p, q):
        # 获取从根节点到p，q的路径，遍历路径查找最近公共祖先
        if not root:
            return None

        path1, path2 = [], []
        self.getPath(root, p, path1)
        self.getPath(root, q, path2)

        res = None
        i = j = 0
        while i < len(path1) and j < len(path2):
            if path1[i] == path2[j]:
                res = path1[i]
            i += 1
            j += 1
        return res

    def getPath(self, root, target, path):
        if not root:
            return False
        path.append(root)
        if root.val == target.val:
            return True
        if (self.getPath(root.left, target, path) or self.getPath(root.right, target, path)):
            return True
        path.pop()