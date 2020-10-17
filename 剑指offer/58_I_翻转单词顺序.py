#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return s
            # # 首先翻转整个句子，然后反转每个单次
            # s = s[::-1]

            # # 翻转单词
            # words_list = s.strip().split()
            # res = []
            # for word in words_list:
            #     res.append(word[::-1])
            # return ' '.join(res)

        # 方法2
        s = s.strip().split()
        s.reverse()
        return ' '.join(s)

obj = Solution()
print(obj.reverseWords("a good   example"))
