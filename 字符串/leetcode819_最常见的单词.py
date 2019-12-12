#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/20 13:39
# Author: Hou hailun

"""
给定一个段落 (paragraph) 和一个禁用单词列表 (banned)。返回出现次数最多，同时不在禁用列表中的单词。题目保证至少有一个词不在禁用列表中，而且答案唯一。
禁用列表中的单词用小写字母表示，不含标点符号。段落中的单词不区分大小写。答案都是小写字母。
示例：
输入:
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
输出: "ball"
解释:
"hit" 出现了3次，但它是一个禁用的单词。
"ball" 出现了2次 (同时没有其他单词出现2次)，所以它是段落里出现次数最多的，且不在禁用列表中的单词。
注意，所有这些单词在段落里不区分大小写，标点符号需要忽略（即使是紧挨着单词也忽略， 比如 "ball,"），
"hit"不是最终的答案，虽然它出现次数更多，但它在禁用单词列表中。
"""
import re
from collections import Counter


class Solution:
    def mostCommonWord(self, paragraph, banned):
        """
        :type paragraph: str
        :type banned: List[str]
        :rtype: str
        """
        paragraph = paragraph.lower()
        word_list = re.findall(r'[a-zA-z]+', paragraph)

        # 利用Counter.most_common()选取指定最大次数的数据
        word_count = Counter(word_list)
        res = word_count.most_common(len(banned) + 1)
        for word_cnt in res:
            if word_cnt[0] not in banned:
                return word_cnt[0]


obj = Solution()
print(obj.mostCommonWord("b,b,b,c", ['a']))