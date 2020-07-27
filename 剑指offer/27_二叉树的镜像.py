#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 请完成一个函数，输入一个二叉树，该函数输出它的镜像。
#
# 例如输入：
#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# 镜像输出：
#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1
#
# 示例 1：
# 输入：root = [4,2,7,1,3,6,9]
# 输出：[4,7,2,9,6,3,1]


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def mirrorTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # 左右孩子交换
        if root is None:
            return None

        root.left, root.right = root.right, root.left
        self.mirrorTree(root.left)
        self.mirrorTree(root.right)
        return root