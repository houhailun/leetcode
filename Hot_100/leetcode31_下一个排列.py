#!/usr/bin/env python
# -*-encoding:utf-8 -*-


"""
实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须原地修改，只允许使用额外常数空间。

以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

此题的目的是求一组元素可以组成的所有数字中比这组元素组成的数字下一大的一组序列

1.一种特殊情况：当序列的元素递减的时候肯定是不存在比它大的序列了，像[3,2,1]组成的数字321已经是最大的了
2.当不是上面的特殊情况的时候，举个例子：
    [1,3,2,4]的下一大序列是[1,3,4,2]
    [1,3,4,2]的下一大序列是[1,4,2,3]
    [1,4,3,2]的下一大序列是[2,1,3,4]
从上面，我们可以发现规律，从序列的后面向前面看，如果nums[i]>nums[i-1]那么这个序列就存在下一大元素
    a.当序列的最后两个元素满足nums[i]>nums[i-1],那么直接交换位置就可以了，像[1,3,2,4]-->[1,3,4,2]
    b.当序列是最后两个元素之前的元素满足nums[i]>nums[i-1]，那么我们就要考虑几个问题了，像[1,3,4,2]--》[1,4,2,3]
        在[1,3,4,2]中，从后向前遍历，3和4满足条件，交换他们之后还要对i和之后元素进行排序，不然得到的就是[1,4,3,2]
        在[1,4,3,2]中，1和4满足条件，但是我们不能直接交换他们，我们要在i之后的序列中找一个满足大于i-1位置元素的最小元素和它交换位置
"""


class Solution:
    def nextPermutation(self, nums):
        if not nums:
            return None

        flag = False
        for i in range(len(nums) - 1, 0, -1):
            # 存在下一个大的排列
            if nums[i] > nums[i - 1]:
                s = sorted(nums[i:])  # 对i后面的元素排序
                if len(s) > 0:
                    # 1、在s中找到最小的大于i-1位置的元素
                    ss = [j for j in s if j > nums[i - 1]][0]

                    # 2、在nums中找到下标，和i-1元素交换位置
                    ix = nums.index(ss, i)
                    nums[i - 1], nums[ix] = nums[ix], nums[i - 1]

                    # 3、交换后把i位置后的元素排序
                    nums[i:] = sorted(nums[i:])
                # 最后元素满足nums[i] > nums[i-1], 直接交换
                else:
                    nums[i], nums[i - 1] = nums[i - 1], nums[i]
                flag = True
                return nums

        if not flag:
            return nums.sort()


if __name__ == "__main__":
    obj = Solution()
    print(obj.nextPermutation([3,2,1]))