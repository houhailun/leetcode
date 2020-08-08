#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def singleNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 方法1：哈希表，空间复杂度不满足O(1)

        # 方法：本题是“数组中有一个数字出现1次，其他出现2次”的扩展
        # 空间复杂度也不满足O(1)
        if not nums:
            return []
        # step1：异或数组中全部元素，最后数字res必然不为0
        res = 0
        for num in nums:
            res ^= num

        # step2: 确认res二进制中为1的位数，来区分所求的两个数字
        bin_ix = 0x01
        while res & bin_ix == 0:
            bin_ix = bin_ix << 1

        # step3: 根据bin_ix是否为1，把数组分位2个子数组，这个每个子数组中必然有“1个数出现1次，其他数出现2次的情况”
        # res1, res2 = [], []
        # for num in nums:
        #     if num & bin_ix > 0:
        #         res1.append(num)
        #     else:
        #         res2.append(num)

        # # step4：在每个子数组中求出现1次的数字
        # num1 = num2 = 0
        # for num in res1:
        #     num1 ^= num
        # for num in res2:
        #     num2 ^= num

        # 优化，直接在原址上遍历
        res1 = res2 = 0
        for num in nums:
            if num & bin_ix:
                res1 ^= num
            else:
                res2 ^= num
        return [res1, res2]


obj = Solution()
print(obj.singleNumbers([4,1,4,6]))