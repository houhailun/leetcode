#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/14 14:38
# Author: Hou hailun

# 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
# 你可以假设数组是非空的，并且给定的数组总是存在多数元素。

# 多数即出现次数超过一半的元素
# 思路1：排序后找中位数即可，时间复杂度O(logN)
# 思路2：记录元素次数，若相同元素次数加1，不同元素次数减1，多数必然是次数大于1的数字


class Solution:
    def majorityElement(self, nums):
        if not nums:
            return

        # 方法1：排序后取中位数
        return self.majorityElement_v1(nums)

        # 方法2：比较元素次数
        return self.majorityElement_v2(nums)

        # 方法3：利用collections模块
        return self.majorityElement_v3(nums)

    def majorityElement_v1(self, nums):
        nums.sort()
        return nums[len(nums)//2]

    def majorityElement_v2(self, nums):
        cnt = 1
        ret = nums[0]
        for num in nums[1:]:
            if num == ret:
                cnt += 1
            else:
                cnt -= 1
            if cnt <= 0:
                ret = num
        return ret

    def majorityElement_v3(self, nums):
        from collections import Counter
        c = Counter(nums)
        return c.most_common(1)[0][0]  # [(num, cnt)]


obj = Solution()
a = [3,2,3]
b = [1,2,2,2,2,1,1,1,2,2]
print(obj.majorityElement(a))
print(obj.majorityElement(b))