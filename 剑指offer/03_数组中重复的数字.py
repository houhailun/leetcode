#!/usr/bin/env python
# -*- encoding:utf-8 -*-


"""
找出数组中重复的数字。
在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
示例 1：
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3
"""


class Solution(object):
    def findRepeatNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return None

        # 方法1：python的count方法
        # return self.findRepeatNumber_core1(nums)

        # 方法2：利用哈希表
        # return self.findRepeatNumber_core2(nums)

        # 方法3: 利用数组长度为n，元素为0~n-1的性质
        # 具体做法就是因为题目中给的元素是 < len（nums）的，所以我们可以让位置i 的地方放元素i。
        # 如果位置i的元素不是i的话，那么我们就把i元素的位置放到它应该在的位置，
        # 即 nums[i] 和nums[nums[i]]的元素交换，这样就把原来在nums[i]的元素正确归位了。
        # 如果发现 要把 nums[i]正确归位的时候，发现nums[i]（这个nums[i]是下标）那个位置上的元素和要归位的元素已经一样了，说明就重复了，重复了就return
        return self.findRepeatNumber_core3(nums)

    def findRepeatNumber_core1(self, nums):
        # 时间复杂度为O(n*n)
        for num in nums:
            if nums.count(num) > 1:
                return num

    def findRepeatNumber_core2(self, nums):
        # 时间复杂度为O(n), 空间复杂度为O(n)
        hash_table = [0] * len(nums)
        for num in nums:
            if hash_table[num] > 0:
                return num
            hash_table[num] += 1
        print(hash_table)

    def findRepeatNumber_core3(self, nums):
        for i in range(len(nums)):
            while i != nums[i]:  # i位置不等于nums[i]
                if nums[i] == nums[nums[i]]:  # 在nums[i]位置上已经存在元素，则表示nums[i]元素是重复的
                    return nums[i]
                # 把i位置的元素num[i]当道nums[nums[i]]位置上
                temp = nums[i]
                nums[i], nums[temp] = nums[temp], nums[i]


if __name__ == "__main__":
    obj = Solution()
    print(obj.findRepeatNumber([2,3,1,0,2,5,3]))