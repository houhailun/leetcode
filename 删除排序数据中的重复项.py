#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目名称：删除排序数据中的重复项

题目描述：给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
示例 1:
给定数组 nums = [1,1,2],
函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。
你不需要考虑数组中超出新长度后面的元素。
示例 2:
给定 nums = [0,0,1,1,1,2,2,3,3,4],
函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
你不需要考虑数组中超出新长度后面的元素。
说明:
为什么返回数值是整数，但输出的答案是数组呢?
请注意，输入数组是以“引用”方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
你可以想象内部操作如下:
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);
// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);

解题思路：要求在排序数组上原址去重，空间复杂度为O(1)
    1、检查nums[i] == nums[i+1]:删除nums[i+1],直到不重复未知
    2、nums[i] != nums[i+1]:i+=1
"""


class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 用时156ms，pop（i）时间复杂度为O（n）
        if not nums:
            return 0
        for i in range(len(nums)):
            while i < len(nums)-1 and nums[i] == nums[i+1]:
                nums.pop(i+1)

        return len(nums)

    def removeDuplicates_v2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 用时156ms
        if not nums:
            return 0
        # 优化：从1开始逐个和前面比较，相同则pop(i-1),,避免每次都要检查下标越界
        # 问题：下标越界，pop/del后列表长度减1，此时range()会根据第一次返回的迭代，造成越界问题
        for i in range(len(nums)-1):
            print('-------------------------')
            print('i:%d, nums:%s, len(nums):%d' % (i, nums, len(nums)))
            while nums[i] == nums[i+1]:
                del nums[i+1]

        return len(nums)

    # 别人的代码：并没有真实删除，只是把后面不重复元素复制到前面
    def removeDuplicates_v3(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        length = len(nums)
        if length <= 1:
            return length
        # 两个指针:fast在前，slow在后,fast指向和slow不相等的下表，然后复制
        slow = 0
        for fast in range(length - 1):
            if nums[fast] != nums[fast + 1]:
                nums[slow + 1] = nums[fast + 1]
                slow += 1
        return slow + 1


cls = Solution()
a = [1,2,2,3]
print(cls.removeDuplicates_v3(a))
print(a)
