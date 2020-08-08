#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution:
    def isBalanced(self, root):
        def helper(root):
            if not root:
                return 0
            # 方法2：后续遍历，从下往上判断，如果某个子树不平衡，那么整个树不平衡
            left = helper(root.left)
            right = helper(root.right)
            if left == -1 or right == -1:
                return -1
            return -1 if abs(left-right)>1 else max(left, right)+1
        return helper(root) != -1