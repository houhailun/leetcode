#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
#
# 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
# 但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
#     1
#    / \
#   2   2
#    \   \
#    3    3
#
# 示例 1：
# 输入：root = [1,2,2,3,4,4,3]
# 输出：true
# 示例 2：
# 输入：root = [1,2,2,null,3,null,3]
# 输出：false


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        # 方法：对称二叉树即节点的左孩子和右孩子相等
        # 从根节点开始，采用前序遍历方法
        if root is None:  # 空树为对称
            return True
        return self.helper(root.left, root.right)

    def helper(self, root1, root2):
        # 两节点都没有，表示遍历完成
        if root1 is None and root2 is None:
            return True
        # 其中1个没有，表示非对称
        if root1 is None or root2 is None:
            return False
        # 值不相等，非对称
        if root1.val != root2.val:
            return False
        return self.helper(root1.left, root2.right) and self.helper(root1.right, root2.left)