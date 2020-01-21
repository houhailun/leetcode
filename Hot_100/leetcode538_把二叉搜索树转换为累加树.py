#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/16 16:58
# Author: Hou hailun

"""
给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。
例如：
输入: 二叉搜索树:
              5
            /   \
           2     13

输出: 转换为累加树:
             18
            /   \
          20     13
"""

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    num = 0

    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # 思路：因为二叉搜索树的性质：跟节点值大于其左孩子节点值，小于其有孩子节点值
        #   可以从右节点开始遍历(右节点已经是最大值节点),依循右-根-左的顺序遍历，依次把当前节点值和上一个节点值相加，得到当前节点的累加值
        if root is None:
            return

        self.convertBST(root.right)

        root.val = root.val + self.num
        self.num = root.val

        self.convertBST(root.left)
        return root