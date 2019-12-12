#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目描述：对称二叉树
给定一个二叉树，检查它是否是镜像对称的。
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3

解题思路：对称二叉树：节点的左孩子等于其兄弟节点的右孩子
    方法：检查根节点是否存在，val是否相等，相等后，递归遍历其左子树和右子树
"""


class Solution:
    def isSymmetric(self, root):
        if not root:  # 空树是对称的
            return True

        return self.helper(root.left, root.right)

    def helper(self, l_root, r_root):
        if not l_root and not r_root:
            return True
        if not l_root or not r_root:
            return False
        if l_root.val != r_root.val:
            return False
        return self.helper(l_root.left, r_root.right) and self.helper(l_root.right, r_root.left)