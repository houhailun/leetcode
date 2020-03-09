#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 16:23
# Author: Hou hailun

# 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
# 示例 1：
# 输入：[3,4,5,1,2]
# 输出：1
# 示例 2：
# 输入：[2,2,2,0,1]
# 输出：0

# 解题思路：数组为递增，旋转后为2个递增数组，在递增数组中查找值，可以用二分法查找；本体属于二分法的变形
# 1、判断中间值属于哪个子数组：
#   case1：mid_val > low_val -> 前面的子数组，则在后面的子数组中查找


class Solution(object):
    def minArray(self, numbers):
        """
        :type numbers: List[int]
        :rtype: int
        """
        if not numbers:
            return None

        # 方法1：暴力法
        # return self.min_array_helper1(numbers)

        # 方法2：
        # return self.min_array_helper2(numbers)

        # 方法3：二分法
        return self.min_array_helper3(numbers)

    def min_array_helper1(self, numbers):
        # 顺序遍历数组，找最小值
        min_val = numbers[0]
        for val in numbers:
            if min_val > val:
                min_val = val
        return min_val

    def min_array_helper2(self, numbers):
        # 优化：由于数组是递增旋转，则构成2个递增的子数组，最小数字必然是第一个比前数字小的
        min_val = numbers[0]
        for i in range(1, len(numbers)):
            if numbers[i] < numbers[i-1]:
                min_val = numbers[i]
        return min_val

    def min_array_helper3(self, numbers):
        if len(numbers) == 0:
            return None
        s = 0
        e = len(numbers) - 1
        if numbers[s] < numbers[e]:
            return numbers[s]

        while s < e - 1:
            index = (s + e) // 2
            if numbers[s] == numbers[e] == numbers[index]:
                last = numbers[0]
                for each in numbers[s:e + 1]:
                    if each < last:
                        return each
                    last = each
            if numbers[index] >= numbers[s]:
                s = index
            else:
                e = index
        return numbers[e]


obj = Solution()
a = [5, 5, 5, 1, 2]
print(obj.minArray(a))