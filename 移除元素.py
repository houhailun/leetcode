#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
题目名称：移除元素

题目描述：给定一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，返回移除后数组的新长度。
不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

示例 1:
给定 nums = [3,2,2,3], val = 3
函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
你不需要考虑数组中超出新长度后面的元素。
示例 2:
给定 nums = [0,1,2,2,3,0,4,2], val = 2,
函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
注意这五个元素可为任意顺序。
你不需要考虑数组中超出新长度后面的元素。
说明:
为什么返回数值是整数，但输出的答案是数组呢?
请注意，输入数组是以“引用”方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
你可以想象内部操作如下:
// nums 是以“引用”方式传递的。也就是说，不对实参作任何拷贝
int len = removeElement(nums, val);
// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);

解题思路：返回的是去除val的列表长度，根据题意可以不必要删除，只需要把前k个元素为非val即可
    1、初始slow = fast = 0
    当 nums[j] 与给定的值相等时，递增 j 以跳过该元素。只要 nums[j] =val，我们就复制 nums[j] 到 nums[i] 并同时递增两个索引。
    重复这一过程，直到 j 到达数组的末尾，该数组的新长度为 i。
"""


class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if not nums:
            return 0
        slow = fast = 0
        length = len(nums)
        while fast < length:
            if nums[fast] == val:
                fast += 1
            else:
                nums[slow] = nums[fast]
                slow += 1
                fast += 1
        return slow

    def removeElement_v2(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        while val in nums:
            nums.remove(val)
        return len(nums)

cls = Solution()
a = [0,1,2,2,3,0,4,2]
print(cls.removeElement_v2(a, 2))
print(a)
