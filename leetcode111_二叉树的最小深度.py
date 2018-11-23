#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明: 叶子节点是指没有子节点的节点。

示例:

给定二叉树 [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回它的最小深度  2.

解题思路:注意题目隐含条件(根节点到叶子节点的最短路径的节点数据，如果某个节点只有一个孩子节点，那么该节点的最小深度等于孩子节点+1)
    方法1：和最大深度类似，递归实现
    方法2：层析遍历，如果某节点左右子树都没有，则退出
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        # 左右子节点都为空：只有根节点或者叶子节点
        if not root.left and not root.right:
            return 1
        # 只有右孩子，则深度为右孩子节点深度+1
        if not root.left and root.right:
            return self.minDepth(root.right)+1
        if root.left and not root.right:
            return self.minDepth(root.left)+1
        return min(self.minDepth(root.left), self.minDepth(root.right))+1

    def min_depth(self, root):
        if not root:
            return 0
        depth = 1
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.left and not node.right:
                return depth
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

            depth += 1
        return depth