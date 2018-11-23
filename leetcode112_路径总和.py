#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
模块描述: 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
说明: 叶子节点是指没有子节点的节点。
示例:
给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。

解题思路: 从根节点开始中序遍历二叉树
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def __init__(self):
        self.tree = None

    def hasPathSum(self, root, sum):
        if root is None:
            return False
        # 叶子节点并且当前节点val=sum，表示True
        if root.left is None and root.right is None and sum == root.val:
            return True
        return self.hasPathSum(root.left, sum-root.val) or self.hasPathSum(root.right, sum-root.val)

    def create_tree(self, root, list, i):
        # 层次遍历建立
        if i < len(list):
            if list[i] == '#':
                return None
            root = TreeNode(list[i])
            root.left = self.create_tree(root.left, list, 2*i+1)
            root.right = self.create_tree(root.left, list, 2*i+2)
        return root

    def pre_order(self, root):
        print(root.val)
        if root.left:
            self.pre_order(root.left)
        if root.right:
            self.pre_order(root.right)


cls = Solution()
tree = cls.create_tree(None, [5,4,8,11,'#',13,4,7,2,'#','#','#',1], 0)
# cls.pre_order(tree)
print(cls.hasPathSum(tree, 22))
