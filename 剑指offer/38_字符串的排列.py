#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution:
    def permutation(self, s):
        # c, res = list(s), []
        #
        # def dfs(x):
        #     if x == len(c) - 1:
        #         res.append(''.join(c)) # 添加排列方案
        #         return
        #     dic = set()
        #     for i in range(x, len(c)):
        #         if c[i] in dic: continue # 重复，因此剪枝
        #         dic.add(c[i])
        #         c[i], c[x] = c[x], c[i] # 交换，将 c[i] 固定在第 x 位
        #         dfs(x + 1) # 开启固定第 x + 1 位字符
        #         c[i], c[x] = c[x], c[i] # 恢复交换
        # dfs(0)
        # return res
        ans = ['']
        for word in s:
            ans = [p[:i] + word + p[i:]
                   for p in ans for i in range((p + word).index(word) + 1)]
        return ans


obj = Solution()
print(obj.permutation("abc"))
