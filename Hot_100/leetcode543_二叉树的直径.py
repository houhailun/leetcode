#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/20 10:48
# Author: Hou hailun

# 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。


class Solution:

    def __init__(self):
        self.max = 0

    def diameterOfBinaryTree(self, root):
        self.depth(root)

        return self.max

    def depth(self, root):
        if not root:
            return 0
        l = self.depth(root.left)
        r = self.depth(root.right)
        '''每个结点都要去判断左子树+右子树的高度是否大于self.max，更新最大值'''
        self.max = max(self.max, l + r)

        # 返回的是高度
        return max(l, r) + 1
