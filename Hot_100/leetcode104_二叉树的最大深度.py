#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 14:03
# Author: Hou hailun

# 深度：从根节点到叶子节点的最长路径上的节点数
# 二叉树的最大深度 = max(左子树的深度，右子树的深度) + 1


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 思路：属于前序遍历变形，因为是从根节点开始遍历，二叉树的最大深度等于其左右子树的最大深度+1
        # 方法一：递归版本
        # if root is None:
        #     return 0
        #
        # return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

        # 方法二：迭代版本, 层序遍历
        if not root:
            return 0
        ans, count = [root], 1
        while ans:
            n = len(ans)
            for i in range(n):
                r = ans.pop(0)
                if r:
                    if not r.left and not r.right:
                        return count
                    ans.append(r.left if r.left else [])
                    ans.append(r.right if r.right else [])
            count += 1
        return count