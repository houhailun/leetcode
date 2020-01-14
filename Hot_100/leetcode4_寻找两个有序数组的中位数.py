#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 13:07
# Author: Hou hailun

"""
给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。
示例 1:
nums1 = [1, 3]
nums2 = [2]
则中位数是 2.0
示例 2:
nums1 = [1, 2]
nums2 = [3, 4]
则中位数是 (2 + 3)/2 = 2.5
"""


class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        # 方法一：把两个数组拼接为1个大有序数组，然后返回中位数
        # 时间复杂度O(m+n), 空间复杂度O(m+n)
        # ret = list()
        # i = j = 0
        # while i < len(nums1) and j < len(nums2):
        #     if nums1[i] <= nums2[j]:
        #         ret.append(nums1[i])
        #         i += 1
        #     else:
        #         ret.append(nums2[j])
        #         j += 1
        #
        # ret.extend(nums1[i:])
        # ret.extend(nums2[j:])
        # print(ret)
        #
        # if len(ret) % 2 == 1:
        #     return ret[len(ret) // 2]
        # return (ret[len(ret)//2-1] + ret[len(ret)//2]) / 2

        # 方法二：利用哈希表，数值作为下标
        # 缺点：数据不能有重复数据
        # hash_table = {}
        # for num in nums1:
        #     hash_table[num] = 1
        # for num in nums2:
        #     hash_table[num] = 1
        # print(hash_table)
        #
        # ret = list(hash_table.keys())
        # ret.sort()
        # if len(ret) % 2 == 1:
        #     return ret[len(ret) // 2]
        # return (ret[len(ret)//2-1] + ret[len(ret)//2]) / 2

        # 方法三：因为两个数组是有序的，因此可以用二分法查找，这个题目可以归纳为：寻找第K大的数字
        # TODO: 暂时没想出好方法




obj = Solution()
nums1 = [1, 1]
nums2 = [1, 2]
print(obj.findMedianSortedArrays(nums1, nums2))