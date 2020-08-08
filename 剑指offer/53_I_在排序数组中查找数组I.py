class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # 利用python的语法
        # return nums.count(target)

        # 有序数组查找元素 -> 二分法
        # 最坏情况下O(N)
        if not nums:
            return 0
        # start, end = 0, len(nums)-1
        # while start <= end:
        #     mid = (start + end) // 2
        #     if nums[mid] == target:
        #         left = right = mid
        #         while left > start and nums[left-1] == target:
        #             left -= 1
        #         while right < end and nums[right+1] == target:
        #             right += 1
        #         return right - left + 1
        #     elif nums[mid] < target:
        #         start = mid + 1
        #     else:
        #         end = mid - 1
        # return 0

        # 方法2：二分, O(N*logN)
        first_ix = self.getFirst(nums, target)
        last_ix = self.getLast(nums, target)
        return last_ix - first_ix + 1 if first_ix>-1 and last_ix>-1 else 0

    def getFirst(self, nums, target):
        start, end = 0, len(nums)-1
        while start <= end:
            mid = (start + end) // 2
            if nums[mid] == target:
                if mid == 0 or nums[mid-1] != target:
                    return mid
                else:
                    end = mid - 1
            elif nums[mid] < target:
                start = mid + 1
            else:
                end = mid - 1
        return -1

    def getLast(self, nums, target):
        start, end = 0, len(nums)-1
        while start <= end:
            mid = (start + end) // 2
            if nums[mid] == target:
                if mid == end or nums[mid+1] != target:
                    return mid
                else:
                    start = mid + 1
            elif nums[mid] < target:
                start = mid + 1
            else:
                end = mid - 1
        return -1