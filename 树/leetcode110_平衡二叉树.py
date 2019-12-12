#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。

示例 1:
给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
返回 true 。

示例 2:
给定二叉树 [1,2,2,3,3,null,null,4,4]
       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。

解题思路：
    方法1：根据二叉树的深度来检查是否是平衡二叉树,从根节点开始计算左右子树的深度，比较是否时平衡，递归检查左子树、右子树是否时平衡
    方法2：方法1有个缺点就是每次从上往下检查是否平衡时，都重复的计算了子节点，那么我们可以考虑从下往上进行判断，当某个子树不平衡则整个二叉树不平衡，
        子树平衡则往上继续判断
        1、后续遍历到最底层节点，获取该节点左右子树的深度，这样每个节点只需要遍历依次即可
        2、不平衡则直接返回False
        3、平衡则继续
"""


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def is_balance_tree(self, root):
        if not root:
            return True

        # 左子树节点深度和右子树深度差大于1
        if abs(self.tree_depth(root.left)-self.tree_depth(root.right)) > 1:
            return False
        return self.is_balance_tree(root.left) and self.is_balance_tree(root.right)

    def tree_depth(self, root):
        # 求节点root的深度
        if not root:
            return 0
        return max(self.tree_depth(root.left), self.tree_depth(root.right))+1

    def is_balance_tree_v2(self, root):
        if not root:
            return True
        self.flag = True

        def max_depth(root):
            if not root:
                return 0
            left = max_depth(root.left)
            right = max_depth(root.right)
            if abs(left - right) > 1:
                self.flag = False
            return max(left, right) + 1

        max_depth(root)
        return self.flag == True
