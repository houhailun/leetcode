#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
leetcode刷题
"""

"""
题目1：两数之和
描述：给定一个整数数组和一个目标值，找出数组中和为目标值的两个数。你可以假设每个输入只对应一种答案，且同样的元素不能被重复利用。

示例:给定 nums = [2, 7, 11, 15], target = 9   因为 nums[0] + nums[1] = 2 + 7 = 9  所以返回 [0, 1]
"""

class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums:
            return []
        '''
        # 方法1：暴力法，两次循环, O(n*n)
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i]+nums[j] == target:
                    return [i, j]
        return []
        '''

        # 方法2：两次遍历哈希表
        # 空间换取时间，第一次：把每个元素的值和索引添加到表中，第二次：检查每个元素所对应的目标元素(target-nums[i])是否在表中
        '''
        hash_table = {}
        for ix, num in enumerate(nums):
            hash_table[ix] = num

        print(hash_table)
        for ix, num in hash_table.items():
            if target-nums[ix] in nums:
                if nums.index(target-nums[ix]) != ix:  # 不能是同一个数字
                    return [ix, nums.index(target-nums[ix])]
        return []
        '''

        # 方法3：遍历一次哈希表
        # 在迭代并把元素插入到哈希表时，可以检查哈希表中是否已经存在当前元素对应的目标元素
        hash_table = {}
        for ix, num in enumerate(nums):
            res = target-num
            if res in hash_table.values():
                res_ix = nums.index(res)
                if res_ix != ix:
                    return [res_ix, ix]
            hash_table[ix] = num
        return []


if __name__ == "__main__":
    cls = Solution()
    print(cls.twoSum([3, 2, 4], 6))

