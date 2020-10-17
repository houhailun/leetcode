#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def oneEditAway(self, first, second):
        """
        :type first: str
        :type second: str
        :rtype: bool
        """
        # 一次或0次编辑距离: 字符串要不相等，要不只差一位(可能是长度相差一位，可能是对应位上字符不同)
        # 左右指针问题,确定不匹配位置first的是[i,j], second的是[i,k],如果区间大于1，则说明不是一次编辑
        # case: 字符串相等，0次编辑距离
        if first == second:
            return True
        len1, len2 = len(first)-1, len(second)-1
        i = 0
        j, k = len1, len2
        # 从头遍历，确定不匹配的位置
        while i <= len1 and i <= len2 and first[i] == second[i]:
            i += 1

        # 从尾部遍历，确定不匹配的位置
        while j >= 0 and k >= 0 and first[j] == second[k]:
            j -= 1
            k -= 1

        return j - i < 1 and k - i < 1

obj = Solution()
print(obj.oneEditAway("ab", "bc"))