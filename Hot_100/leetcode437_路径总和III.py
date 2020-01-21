#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/16 17:27
# Author: Hou hailun

"""
给定一个二叉树，它的每个结点都存放着一个整数值。
找出路径和等于给定数值的路径总数。
路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。
示例：
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1
返回 3。和等于 8 的路径有:
1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11
"""

"""
首先解读题干，题干的要求是找和为sum的路径总数，这次路径的起点和终点不要求是根结点和叶结点，可以是任意起终点，而且结点的数值有正有负，但是要求不能回溯，只能是从父结点到子结点的。在已经做了路径总和一和二的基础上，我们用一个全局变量来保存路径总数量，在主调函数中定义变量self.result=0。
因为数值有正有负，所以在当我们找到一条路径和已经等于sum的时候，不能停止对这条路径的递归，因为下面的结点可能加加减减，再次出现路径和为sum的情况，因此当遇到和为sum的情况时，只需要用self.result+=1把这条路径记住，然后继续向下进行即可。即下面这段代码：

由于路径的起点不一定是根结点，所以需要对这棵树的所有结点都执行一次搜索，就是树的遍历问题，每到一个结点就执行一次dfs去搜索以该结点为起点的路径：
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):
    def __init__(self):
        self.path_sum = 0

    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        # 因为路径方向是从上往下，所以使用前序遍历
        if root is None:
            return self.path_sum

        self.getPathSum(root, sum)
        self.getPathSum(root.left, sum)
        self.getPathSum(root.right, sum)

    def getPathNum(self, root, sum):
        """
        依据当前树找目标值，进而找到路径数量
        :param root:
        :param sum:
        :return:
        """
        if root is None:
            return
        if root.val == sum:
            self.path_sum += 1
        new_sum = sum - root.val
        self.getPathNum(root.left, new_sum)
        self.getPathNum(root.right, new_sum)