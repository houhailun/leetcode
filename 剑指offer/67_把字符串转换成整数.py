#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution(object):
    def strToInt(self, str):
        """
        :type str: str
        :rtype: int
        """
        if not str:
            return 0
        str = str.strip()
        res = ''
        flag = ''
        for i, ch in enumerate(str):
            # case: 开头无效字符，直接返回
            if i == 0 and ch not in ['-', '+'] and not ch.isdigit():
                return 0

            # case: 开头是正负号，则记录标记，跳过
            if i == 0 and ch in ['-', '+']:
                flag = ch
                continue

            # case: 中间有正负号
            if ch in ['-', '+']:
                return 0

            # case: 中间有非数字字符,跳出
            if not ch.isdigit():
                break
            res += ch

        result = 0
        for ch in res:
            result = result * 10 + ord(ch) - ord('0')
        if flag == '+':
            return result
        elif flag == '-':
            return -result
        return result

obj = Solution()
print(obj.strToInt("91283472332"))