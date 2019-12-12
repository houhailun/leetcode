#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/18 17:31
# Author: Hou hailun

"""
给定两个没有重复元素的数组 nums1 和 nums2 ，其中nums1 是 nums2 的子集。找到 nums1 中每个元素在 nums2 中的下一个比其大的值。
nums1 中数字 x 的下一个更大元素是指 x 在 nums2 中对应位置的右边的第一个比 x 大的元素。如果不存在，对应位置输出-1。
示例 1:
输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:
    对于num1中的数字4，你无法在第二个数组中找到下一个更大的数字，因此输出 -1。
    对于num1中的数字1，第二个数组中数字1右边的下一个较大数字是 3。
    对于num1中的数字2，第二个数组中没有下一个更大的数字，因此输出 -1。
示例 2:
输入: nums1 = [2,4], nums2 = [1,2,3,4].
输出: [3,-1]
解释:
    对于num1中的数字2，第二个数组中的下一个较大数字是3。
    对于num1中的数字4，第二个数组中没有下一个更大的数字，因此输出 -1。
"""


class Solution(object):
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # 方法1： 暴力求解, 两次循环遍历O(N*N)
        # if nums1 is None or nums2 is None:
        #     return [-1] * len(nums1)
        #
        # res = [-1] * len(nums1)
        # for i in range(len(nums1)):
        #     ix = nums2.index(nums1[i])
        #     print('---',i, ix)
        #     print('--------------')
        #     for j in range(ix, len(nums2)):
        #         print(j, nums2[j])
        #         if nums2[j] > nums1[i]:
        #             res[i] = nums2[j]
        #             break
        # return res

        # 方法2: 栈实现
        stack = []
        hashmap = {}
        # 遍历nums2，把大值作为val，小值作为key，只要第一个大值
        for num in nums2:
            while stack and stack[-1] < num:
                hashmap[stack.pop()] = num
            stack.append(num)
        print(hashmap)
        return [hashmap.get(x, -1) for x in nums1]

obj = Solution()
ret = obj.nextGreaterElement([4,1,2], [1,3,4,2])
print(ret)