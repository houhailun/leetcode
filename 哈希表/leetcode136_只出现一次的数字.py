#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
题目描述：给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
说明：你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
示例 1:
输入: [2,2,1]
输出: 1
示例 2:
输入: [4,1,2,1,2]
输出: 4

解题思路：
    方法1：使用哈希表，有额外空间复杂度
    方法2：异或（相同为0，不同为1）
    方法3：x=sum(set(nums))设置所有元素出现一次并求和，y=sum(nums)所有元素求和，2*x-y即为只出现一次的数字
"""

class Solution:
    def single_number(self, nums):
        if not nums:
            return 0

        ret = 0
        for val in nums:
            ret ^= val
        return ret

    def single_number_v2(self, nums):
        if not nums:
            return 0
        return 2*sum(set(nums)) - sum(nums)


cls = Solution()
print(cls.single_number_v2([4,1,2,1,2]))