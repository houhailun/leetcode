#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/7/29 18:06
# Author: Hou hailun

# 只需要确定是否存在路径，使得路径和等于指定值
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class Solution:
    def hasPathSum(self, root, target):
        if not root:
            return False
        # 叶子节点
        if not root.left and not root.right and root.val == target:
            return True
        return self.hasPathSum(root.left, target-root.data) or self.hasPathSum(root.right, target-root.data)

    def findPath(self, root, target):
        res, path = [], []

        def recur(_root, _target):
            if not _root:
                return
            path.append(_root.data)
            _target = _target - _root.data
            if target == 0 and not _root.left and not _root.right:
                res.append(path)
            recur(_root.left, _target)
            recur(_root.right, _target)
            path.pop()  # 回溯
        recur(root, target)
        return res
