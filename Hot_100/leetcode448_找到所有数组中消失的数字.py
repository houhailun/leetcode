#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/1/16 17:06
# Author: Hou hailun

"""
给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
找到所有在 [1, n] 范围之间没有出现在数组中的数字。
您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。
示例:
输入:
[4,3,2,7,8,2,3,1]
输出:
[5,6]
"""

# 方法1：两次循环，依次对比从1~n, 判断是否在数组中  时间复杂度：O(N*N)
# 方法2：[i for i in range(1,n+1)], 遍历判断缺失值  时间复杂度：O(N)， 空间复杂度:O(N)
# 方法3: for i in range(1, n+1), 直接在列表中比较缺失值  时间复杂度：O(N)
# 方法4：数学方法，利用set的相减，set([for i in range(1, n+1)]) - set(nums), 即可求得缺失数据
# 方法5：利用数组性质：元素位于1~n之间，如果没有重复元素，那么必然val=nums.index(val),

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # return self.findDisappearedNumbers_v1(nums)
        return self.findDisappearedNumbers_v2(nums)

    def findDisappearedNumbers_v1(self, nums):
        # nums过长会超时
        ret = []
        for i in range(1, len(nums)+1):
            if i not in nums:
                ret.append(i)
        return ret

    def findDisappearedNumbers_v2(self, nums):
        if len(nums) <= 1:
            return []

        return list(set(range(1, len(nums)+1)) - set(nums))


obj = Solution()
print(obj.findDisappearedNumbers([4,3,2,7,8,2,3,1]))