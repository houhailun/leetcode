#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 15:08
# Author: Hou hailun

# 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
# 输入: [0,1,0,3,12]
# 输出: [1,3,12,0,0]


class Solution:
    def moveZeroes(self, nums):
        if nums is None:
            return None

        # 思路1：空间换时间
        # return self.moveZeros_v1(nums)

        # 题目要求：必须在原数组上操作，不能拷贝额外的数组，因此上述方法不适用
        # 思路2：双指针
        return self.moveZeros_v2(nums)

    def moveZeros_v1(self, nums):
        # 利用辅助列表，把非0元素拍到前，0放到后面
        not_zeros = []
        zeros = []
        for num in nums:
            if num == 0:
                zeros.append(num)
            else:
                not_zeros.append(num)
        return not_zeros+zeros

    def moveZeros_v2(self, nums):
        # 定义两个指针i,j，然后遍历数组，i跟j同时往前走，当遇到0时j停下，i继续往前走。当nums[i]不为0时则将num[i]的元素赋给j的位置，j++,nums[i]被赋值为0
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:  # 把i不为0的放到前面，j来指定放置的位置
                nums[j] = nums[i]
                if i != j:  # i!=j，表示当前有0，则把i位置置为0
                    nums[i] = 0
                j += 1

    def moveZeros_v3(self, nums):
        # 思路：从头开始把非0写到前面，后面填充0
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        for i in range(j, len(nums)):
            nums[i] = 0


obj = Solution()
a = [0,1,0,3,12]
print(obj.moveZeroes(a))