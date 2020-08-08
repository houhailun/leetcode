class Solution(object):
    def reversePairs(self, nums):
        length = len(nums)
        if length < 2:
            return 0
        mid = length // 2
        left_arr = nums[:mid]
        right_arr = nums[mid:]
        # 子数组间的逆序对
        result = self.reversePairs(left_arr) + self.reversePairs(right_arr)

        left_arr.sort()
        right_arr.sort()

        left_index=0
        for i in range(len(right_arr)):
            while left_index < len(left_arr) and left_arr[left_index] <= right_arr[i]:
                left_index +=1
            result = result + (len(left_arr) - left_index)
        return result