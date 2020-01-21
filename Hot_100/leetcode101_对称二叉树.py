#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 10:07
# Author: Hou hailun

"""
给定一个二叉树，检查它是否是镜像对称的。
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
    1
   / \
  2   2
   \   \
   3    3
"""

# 镜像就是节点的左右子树等于其兄弟节点的右左子树
# 思路：比较左子树的左孩子和右子树的有孩子  左子树的有孩子和右子树的左孩子


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
        # 空树是镜像
        if not root:
            return True

        # return self.isSymmetric_helper(root.left, root.right)
        return self.isSymmetric_helper_loop(root)

    # 递归版本
    def isSymmetric_helper(self, l_root, r_root):
        if l_root is None and r_root is None:
            return True
        if l_root is None or r_root is None:
            return False
        if l_root.val != r_root.val:
            return False
        return self.isSymmetric_helper(l_root.left, r_root.right) and self.isSymmetric_helper(l_root.right, r_root.left)

    # 循环版本
    def isSymmetric_helper_loop(self, root):
        stack = [(root.left, root.right)]
        while stack:
            l, r = stack.pop()
            # 其中只有1个为空
            if l is None or r is None:
                if l != r:
                    return False
            else:  # 两个都为空，或者都不是空
                if l.val != r.val:
                    return False
                stack.append((l.left, r.right))
                stack.append((r.left, l.right))
        return True
