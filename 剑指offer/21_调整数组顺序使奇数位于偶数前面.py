class Solution(object):
    def exchange(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return nums

        # 方法1：利用2个辅助队列，分别存放奇数和偶数，最后合并
        # 方法2：双指针法
        # p1指向开头，若为奇数则后移，直到偶数位置
        # p2指向末尾，若为偶数则前移，直到奇数为止
        # 交换p1，p2
        first, end = 0, len(nums) - 1
        while first < end:
            # while first < end and nums[first] % 2 == 1:
            # 优化：使用位于
            while first < end and nums[first] & 1 == 1:
                first += 1
            while first < end and nums[end] & 1 == 0:
                end -= 1
            nums[first], nums[end] = nums[end], nums[first]
        return nums