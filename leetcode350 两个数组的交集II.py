#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
给定两个数组，编写一个函数来计算它们的交集
示例 1:
输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2]
示例 2:
输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [9,4]
说明:

输出结果中的每个元素一定是唯一的。
我们可以不考虑输出结果的顺序。
"""


class Solution:
    def intersection(self, num1, num2):
        """
        交两个数组的交集
        :param num1:
        :param num2:
        :return:
        """
        return list(set(num1) & set(num2))


"""
给定两个数组，编写一个函数来计算它们的交集。

示例 1:
输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2,2]
示例 2:
输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [4,9]
说明：
输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
我们可以不考虑输出结果的顺序。
"""


class Solution_ii:
    def intersection(self, nums1, nums2):
        # 和上一题的区别：要求次数和元素在两个数组中出现的次数一致
        # 方法1：1、找num1，num2的公共元素 2、填充公公元素的次数为min(num1.count(x), num2.count(x))
        # if not nums1 or not nums2:
        #     return []
        # ret = []
        # common = list(set(nums1) & set(nums2))
        # for num in common:
        #     ret += [num] * min(nums1.count(num), nums2.count(num))
        # return ret

        # 方法2:利用字典记录nums1中元素的次数，然后再nums2中逐个查找
        dict_num1 = dict()
        for num in nums1:
            dict_num1[num] = dict_num1[num]+1 if num in dict_num1 else 1

        ans = []
        for num in nums2:
            if num in dict_num1 and dict_num1[num] >= 1:
                ans.append(num)
                dict_num1[num] -= 1
        return ans



if __name__ == "__main__":
    cls = Solution_ii()
    print(cls.intersection([1,2,2,1], [2,2]))