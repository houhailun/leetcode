#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 15:11
# Author: Hou hailun

# 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# 例如，给出
# 前序遍历 preorder = [3,9,20,15,7]
# 中序遍历 inorder = [9,3,15,20,7]
# 返回如下的二叉树：
#
#     3
#    / \
#   9  20
#     /  \
#    15   7

# 前序遍历：根-左-右
# 中序遍历：左-根-右
# 后续遍历：左-右-根
# 层次遍历：利用辅助队列事先
# 解题思路：
#   1、在前序列表中找根节点：开头的节点即为根节点，并构建根节点
#   2、在中序列表中找根节点对应的左右子树：根节点左边的是左子树，右边的是右子树
#   3、递归在左子树和右子树中查找构建


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        # 健壮性检查
        if not preorder or not inorder:
            return None

        loc = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        # 注意前序和中序列表的范围
        root.left = self.buildTree(preorder[1: loc + 1], inorder[: loc])
        root.right = self.buildTree(preorder[loc + 1:], inorder[loc + 1:])
        return root