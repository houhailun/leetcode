#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/12/31 15:06
# Author: Hou hailun

"""
给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
示例:
给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
"""


class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 方法一：暴力法
        # return self.twoSum_navie(nums, target)

        # 方法二：利用哈希表，key为下标，值为数据
        return self.twoSum_hash(nums, target)

    def twoSum_navie(self, nums, target):
        # 暴力法：双重循环  时间复杂度为O(N*N)    空间复杂度为O(1)
        if nums is None:
            return None, None

        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                print(i, j)
                if nums[i] + nums[j] == target:
                    return [i, j]
        return None, None

    def twoSum_hash(self, nums, target):
        if nums is None:
            return None, None

        # 构建哈希表
        hash_table = {}
        for ix, num in enumerate(nums):
            hash_table[ix] = num

        # 遍历哈希表
        for ix, num in hash_table.items():
            if target - nums[ix] in nums:
                if nums.index(target - nums[ix]) != ix:  # 不是同一个数字
                        return [ix, nums.index(target-nums[ix])]
        return []

    def twoSum_hash_once(self, nums, target):
        # 遍历一次，在迭代并构建哈希表的同时检查是否有目标元素
        hash_table = {}
        for ix, num in enumerate(nums):
            res = target - num
            if res in hash_table.values():  # res 如果在哈希表中，说明可能存在两数之和等于目标值
                res_ix = nums.index(res)     # 判断是否是由于同一个值
                if res_ix != ix:
                    return [res_ix, ix]
            hash_table[ix] = num
        return []

obj = Solution()
print(obj.twoSum_hash_once([2, 7, 11, 15], 9))
