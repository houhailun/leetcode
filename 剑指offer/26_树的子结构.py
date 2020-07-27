#!/usr/bin/env python
# -*- encoding:utf-8 -*-


# 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
#
# B是A的子结构， 即 A中有出现和B相同的结构和节点值。
#
# 例如:
# 给定的树 A:
#
#      3
#     / \
#    4   5
#   / \
#  1   2
# 给定的树 B：
#
#    4 
#   /
#  1
# 返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值
#
# 示例 1：
# 输入：A = [1,2,3], B = [3,1]
# 输出：false
# 示例 2：
# 输入：A = [3,4,5,1,2], B = [4,1]
# 输出：true


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSubStructure(self, A, B):
        """
        :type A: TreeNode
        :type B: TreeNode
        :rtype: bool
        """
        # 思路：先在树A中查找树B的根节点；然后按照跟-左-右的顺序递归检查，直到树B全部遍历完成，则表示B是A的子结构
        if not A or not B:
            return False

        result = False
        # 在树A中找树B的根节点
        if A.val == B.val:
            result = self.isSubStructure_helper(A, B)
        if not result:
            result = self.isSubStructure(A.left, B)
        if not result:
            result = self.isSubStructure(A.right, B)
        return result

    def isSubStructure_helper(self, A, B):
        if not B:
            return True
        if not A:
            return False
        if A.val != B.val:
            return False
        return self.isSubStructure_helper(A.left, B.left) and self.isSubStructure_helper(A.right, B.right)
