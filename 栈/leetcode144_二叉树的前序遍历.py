#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/18 17:56
# Author: Hou hailun

# 前序遍历: 根-左-右
# 中序遍历: 左-根-右
# 后续遍历: 左-右-根


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 方法1：递归
        # if root is None:
        #     return []
        # ret = []
        # self.preorderTraversal_v1(root, ret)
        # return ret

        # 方法2：使用栈
        res = []
        if root is None:
            return res
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:  # 先判断右节点
                stack.append(node.right)
            if node.left:  # 在判断左节点
                stack.append(node.left)
        return res

    def preorderTraversal_v1(self, root, ret):
        if root is None:
            return
        ret.append(root.val)
        self.preorderTraversal_v1(root.left, ret)
        self.preorderTraversal_v1(root.right, ret)