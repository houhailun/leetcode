#!/usr/bin/env python
# -*- encoding:utf-8 -*-

def lowestCommonAncestor(root, p, q):
    if root.val < p.val and root.val < q.val:
        return lowestCommonAncestor(root.right, p, q)
    if root.val > p.val and root.val > q.val:
        return lowestCommonAncestor(root.left, p, q)
    return root

def lowestCommonAncestor_v2(root, q, p):
    if p.val > q.val:
        p, q = q, p # 保证 p.val < q.val
        while root:
            if root.val < p.val: # p,q 都在 root 的右子树中
                root = root.right # 遍历至右子节点
            elif root.val > q.val: # p,q 都在 root 的左子树中
                root = root.left # 遍历至左子节点
            else:
                break
        return root

