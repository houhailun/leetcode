#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/11/19 18:07
# Author: Hou hailun

"""
返回 A 的最短的非空连续子数组的长度，该子数组的和至少为 K 。
如果没有和至少为 K 的非空子数组，返回 -1 。
示例 1：
输入：A = [1], K = 1
输出：1
示例 2：
输入：A = [1,2], K = 4
输出：-1
示例 3：
输入：A = [2,-1,2], K = 3
输出：3
"""


class Solution(object):
    def shortestSubarray(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        # 遍历记录和为k的连续子数组，最后判断最短的子数组
        # 如何找到和为k的所有连续子数组：每次从上一个子数组开始的下一位置开始查找，因此需要维护head指针
        if A is None:
            return -1

        res = []
        for head in range(len(A)):
            sum = 0
            print('----------')
            for i in range(head, len(A)):
                sum += A[i]
                print(sum)
                if sum >= K:  # 记录当前下标
                    print(head, i)
                    res.append(i-head+1)
                    break
        print(res)
        if not res:
            return -1
        return min(res)

        # 方法2：
        # 1、求前缀数组和S[n], s[0]=0, s[i] = s[0]+...+s[i-1]
        # 2、用双端队列维护一个单调栈，假设i < j < k, 且s[i] > s[j]则
        # a. if s[k] - s[i] >= K →s[k] - s[j] >= k, 因此此时的s[i]可以被丢弃
        # b.有贪心思想可知这个序列不可能以负数或0开头或者结尾，所以如果存在s[k] <= s[j]队尾丢弃
        # c.此过程众不断计算结果
        from collections import deque
        sum = [0]
        for i in range(len(A)):
            sum[i + 1] = sum[i] + A[i]



obj = Solution()
ret = obj.shortestSubarray()
print(ret)