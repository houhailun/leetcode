#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/10 15:52
# Author: Hou hailun

"""
题目名称：最长公共前缀

题目描述：编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串 ""。
示例 1:
输入: ["flower","flow","flight"]
输出: "fl"

示例 2:
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
说明:所有输入只包含小写字母 a-z 。

解题思路：最长的前缀不会超过最短的字符串，所以先找到最短字符串，然后依次比较
    step1:找出长度最短的字符串
    step2:依次与长度最短的字符串比较，最长公共字串为所有比较后的最短公共字串
"""


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        # # 异常检查
        # if strs is None:
        #     return ""
        # if len(strs) == 1:
        #     return strs[0]
        #
        # # 找到最短的字符串，只需要找到最短字符串的长度即可
        # min_str = min([len(x) for x in strs])
        # ix = 0  # 标记公共前缀的字符个数
        # length = len(strs)
        # while ix < min_str:
        #     # 遍历列表，逐个匹配
        #     for i in range(1, length):
        #         if strs[i][ix] != strs[i - 1][ix]:
        #             return strs[0][:ix]
        #     ix += 1
        # return strs[0][:ix]  # 用时48ms

        # 大神的代码  36ms
        shorest = min(strs, key=len)  # 长度最短的字串
        print(shorest)
        for i, ch in enumerate(shorest):
            for other in strs:
                if other[i] != ch:
                    return shorest[:i]
        return shorest


if __name__ == "__main__":
    obj = Solution()
    print(obj.longestCommonPrefix(["flower","flow","flight"]))