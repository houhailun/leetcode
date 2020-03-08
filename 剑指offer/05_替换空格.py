#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
示例 1：
输入：s = "We are happy."
输出："We%20are%20happy."
"""


class Solution(object):
    def replaceSpace(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return None

        # 方法1：使用python字符串的内置函数replace实现
        # return s.replace(' ', '%20')

        # 方法2：双指针法, 原址修改
        return self.replaceSpace_core(s)

    def replaceSpace_core(self, s):
        # 由于python的字符串不能修改，因此需要先转换为字符串列表 - 修改 - 转换为字符串
        s_list = list(s)
        for i in range(len(s_list)):
            if s_list[i] == ' ':
                s_list[i] = '%20'
        return ''.join(s_list)


if __name__ == "__main__":
    obj = Solution()
    print(obj.replaceSpace('hello python '))