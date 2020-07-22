#!/usr/bin/env python
# -*- encoding:utf-8 -*-


# 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
# 找出所有满足条件且不重复的三元组。
# 注意：答案中不可以包含重复的三元组。

class Solution:
    def threeSum(self, nums):
        size = len(nums)
        if not nums or size < 3:
            return []

        # 排序+ 双指针
        res = []
        nums.sort()
        for i in range(size):
            # 当前元素大于0,后面必然不会出现三元素和等于0,直接返回即可
            if nums[i] > 0:
                return res
            # 排序后相邻两数如果相等，则跳出当前循环继续下一次循环，相同的数只需要计算一次
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left, right = i+1, size-1
            while left < right:
                tmp = nums[i] + nums[left] + nums[right]
                if tmp == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    # 重复元素跳过
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif tmp < 0:
                    left += 1
                else:
                    right -= 1
        return res


if __name__ == "__main__":
    obj = Solution()

    nums = [-1, 0, 1, 2, -1, -4]
    print(obj.threeSum(nums))