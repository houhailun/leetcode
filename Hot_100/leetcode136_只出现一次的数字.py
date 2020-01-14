#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 14:19
# Author: Hou hailun

# 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
# 思路1：根据异或性质：相同元素异或为0，不同为1；所有元素依次异或，最后的结果即为所求
# 思路2：hash表


class Solution(object):
    def singleNumber(self, nums):
        print(nums)
        if nums is None:
            print('nums is none')
            return None

        # 方法1：暴力法，时间复杂度较大
        # return self.singleNumber_v1(nums)

        # 方法2: hash表，占用额外空间
        # return self.singleNumber_v2(nums)

        # 方法3：异或性质
        # return self.singleNumber_v3(nums)

        # 方法4：数学公式
        return self.singleNumber_v4(nums, threshold=2)


    def singleNumber_v1(self, nums):
        for num in nums:
            if nums.count(num) == 1:
                return num
        return None

    def singleNumber_v2(self, nums):
        hash_table = {}
        for num in nums:
            if num not in hash_table:
                hash_table[num] = 1
            else:
                hash_table[num] += 1

        for num in nums:
            if hash_table[num] == 1:
                return num
        return None

    def singleNumber_v3(self, nums):
        ret = 0
        for num in nums:
            ret ^= num
        return ret

    def singleNumber_v4(self, nums, threshold):
        # 数学公式，可灵活扩展到：只有1个数字出现1次，其余数字出现N次
        # threshold: 指定其余数字出现的次数
        x = set(nums)
        return (threshold * sum(x) - sum(nums)) / (threshold-1)

obj = Solution()
a = [4, 1, 2, 1, 2]
print(obj.singleNumber(a))
