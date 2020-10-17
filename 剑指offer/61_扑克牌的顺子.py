#!/usr/bin/env python
# -*- encoding:utf-8 -*-

class Solution(object):
    def isStraight(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
            return False

        def helper1(nums):
            numZero = numGap = 0
            nums.sort()
            for i in range(1, len(nums)):
                if nums[i-1] == 0:
                    numZero += 1
                    continue
                if nums[i] - nums[i-1] > 1:
                    numGap += nums[i] - nums[i-1] - 1
            if numGap > numZero:
                return False
            return True
        # return helper1(nums)

        def helper2(nums):
            # 除大小王外，最大牌-最小牌《5,则可以构成顺子
            repeat = list()
            _max, _min = float('-inf'), float('inf')
            for num in nums:
                if num == 0:
                    continue
                _max = max(_max, num)
                _min = min(_min, num)
                if num in repeat:
                    return False
                repeat.append(num)
            return _max - _min < 5

        return helper2(nums)

obj = Solution()
print(obj.isStraight([0,0,1,2,5]))