#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
示例:
给定有序数组: [-10,-3,0,5,9],
一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
      0
     / \
   -3   9
   /   /
 -10  5

解题思路：
    1、二叉搜索树：根节点值大于其左孩子节点值，小于其右孩子节点值；左右子树同样也而是搜索树
    2、平衡树：左右子树的高度差不超过1
    3、二叉搜索树的中序遍历是排序数组
    方法：1、在有序数组中找到根节点：数组的中间值作为根节点，左半子数组作为左子树，右半子数组作为右子树，递归创建
         2、中序遍历创建BST
         3、因为是每次是从数组中获取中间值作为根节点，因此不存在不平衡问题
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def sortedArrayToBST(self, nums):
        # 中序遍历创建二叉搜索树
        if not nums:
            return None
        mid = len(nums) // 2
        root = TreeNode(nums[mid])  # 根节点
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root


cls = Solution()
tree = cls.sortedArrayToBST([-10, -3, 0, 5, 9])
