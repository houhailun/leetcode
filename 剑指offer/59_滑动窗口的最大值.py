#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # res = []
        # if not nums:
        #     return res
        # # 方法1：max()
        # # 时间复杂度:
        # for i in range(0, len(nums)-k+1):
        #     sub_nums = nums[i: i+k]
        #     res.append(max(sub_nums))
        # return res

        # 方法2：
        res = []
        import collections
        deque = collections.deque()  # 双端队列
        n = len(nums)
        # 设置滑动窗口边界，i为左边界，j为有边界，打包形成滑动窗口，然后遍历
        for i, j in zip(range(1 - k, n - k + 1), range(0, n)):
            # 制定规则，始终在deque[0]存储最大值
            # 1. 先看deque对手，i=0时因前面无值没有值可以滑出（所以从i=1开始，此时num[0]就是滑出值）
            # 同时当nums[i-1]滑出窗口时, 即判定条件是deque[0] == nums[i-1], 将队首取出
            if i > 0 and deque[0] == nums[i - 1]:
                deque.popleft()
            # 然后继续判定划入窗口的值是补入队尾还是队首
            # deque不为空时便有队尾，同时满足队尾deque[-1]<num[j]时将nums[j]补在队首
            # 否则直接补在队尾
            while deque and deque[-1] < nums[j]:
                deque.pop()
            deque.append(nums[j])
            if i >= 0:  # res的补入原则
                res.append(deque[0])
        return res