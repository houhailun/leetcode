#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
我们正在玩一个猜数字游戏。 游戏规则如下：
我从 1 到 n 选择一个数字。 你需要猜我选择了哪个数字。
每次你猜错了，我会告诉你这个数字是大了还是小了。
你调用一个预先定义好的接口 guess(int num)，它会返回 3 个可能的结果（-1，1 或 0）：
-1 : 我的数字比较小
 1 : 我的数字比较大
 0 : 恭喜！你猜对了！
示例 :
输入: n = 10, pick = 6
输出: 6
"""


class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 在1~n中找某个关键字，因此采用二分法
        start, end = 1, n
        while True:
            mid = (start + end) / 2
            flag = guess(mid)
            if 0 == flag:
                return mid
            elif -1 == flag:
                end = mid
            else:
                start = mid
            if end - start == 1:
                return end if flag == 1 else start


cls = Solution()
print(cls.guessNumber(10))