#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
题目描述:给定两个二叉树，编写一个函数来检验它们是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。
"""


class Solution:
    def isSameTree(self, p, q):
        # 两个树都为None，表示遍历完成
        if not p and not q:
            return True
        # p 和 q结构不一样
        if not p or not q:
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False