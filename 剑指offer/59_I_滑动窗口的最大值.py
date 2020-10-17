#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/8/9 10:59
# Author: Hou hailun


class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        res = []
        if not nums:
            return res
        # 方法1：暴力法
        # 时间复杂度: O((n-k+1)k) = O(nk), 有(n-k+1)个窗口，每个窗口内最大值用线性遍历O(k)
        for i in range(0, len(nums)-k+1):
            sub_nums = nums[i: i+k]
            res.append(max(sub_nums))
        return res

        # 方法2：设法把华东窗口中的判断最大值O(k)优化为O(1)
        # 考虑到包含min函数的栈，这里可以使用队列实现
        # 初始化： 双端队列 dequedeque ，结果列表 resres ，数组长度 nn ；
        # 滑动窗口： 左边界范围 i \in [1 - k, n + 1 - k]i∈[1−k,n+1−k] ，右边界范围 j \in [0, n - 1]j∈[0,n−1] ；
        #   1. 若 i > 0i>0 且 队首元素 deque[0]deque[0] == 被删除元素 nums[i - 1]nums[i−1] ：则队首元素出队；
        #   2. 删除 dequedeque 内所有 < nums[j]<nums[j] 的元素，以保持 dequedeque 递减；
        #   3. 将 nums[j]nums[j] 添加至 dequedeque 尾部；
        #   4. 若已形成窗口（即 i \geq 0i≥0 ）：将窗口最大值（即队首元素 deque[0]deque[0] ）添加至列表 resres 。
        res = []
        import collections
        deque = collections.deque()  # 双端队列
        n = len(nums)
        # 设置滑动窗口边界，i为左边界，j为有边界，打包形成滑动窗口，然后遍历
        for i, j in zip(range(1-k, n-k+1), range(0, n)):
            # 制定规则，始终在deque[0]存储最大值
            # 1. 先看deque对手，i=0时因前面无值没有值可以滑出（所以从i=1开始，此时num[0]就是滑出值）
            # 同时当nums[i-1]滑出窗口时, 即判定条件是deque[0] == nums[i-1], 将队首取出
            if i > 0 and deque[0] == nums[i-1]:
                deque.popleft()
            # 然后继续判定划入窗口的值是补入队尾还是队首
            # deque不为空时便有队尾，同时满足队尾deque[-1]<num[j]时将nums[j]补在队首
            # 否则直接补在队尾
            while deque and deque[-1] < nums[j]:
                deque.pop()
            deque.append(nums[j])
            if i >= 0:  #  res的补入原则
                res.append(deque[0])
        return res

obj = Solution()
print(obj.maxSlidingWindow([5,3,4], 2))