#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/7/15 13:12
# Author: Hou hailun


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


# 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标
# 你不能重复利用这个数组中同样的元素
def two_sum(nums, target):
    if not nums:
        return []

    # 暴力法：双重循环O(N*N)
    def two_sum_bp():
        length = len(nums)
        for i in range(length):
            for j in range(i+1, length):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []
    # return two_sum_bp()

    # 哈希表法: 2次遍历
    def two_sum_hash():
        hash_table = {}
        for i, num in enumerate(nums):
            hash_table[i] = num
        for i, num in hash_table.items():
            if target - num in nums:
                if nums.index(target-num) != i:
                    return [i, nums.index(target-num)]
        return []
    # return two_sum_hash()

    # 哈希表: 1次遍历
    def two_sum_hash_once():
        hash_table = {}
        for ix, num in enumerate(nums):
            res = target - num
            if res in hash_table.values():
                res_ix = nums.index(res)
                if res_ix != ix:
                    return [ix, res_ix]
            hash_table[ix] = num
        return []

    return two_sum_hash_once()

# print(two_sum([2, 7, 11, 15], 9))


# 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
# 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
def two_sum_link(l1, l2):
    new_head = ListNode(-1)
    cur_node = new_head
    carry = 0
    while l1 or l2:
        x = l1.val if l1 else 0
        y = l2.val if l2 else 0
        res = x + y + carry

        carry = res // 10
        node = ListNode(res % 10)
        cur_node.next = node
        cur_node = cur_node.next

        if l1:
            l1 = l1.next
        if l2:
            l2 = l2.next
    if carry > 0:  # 最高位有进位
        node = ListNode(carry)
        cur_node.next = node
    return new_head.next


# 两数相加，不用加法
def two_num_sum(num1, num2):
    while num2 > 0:
        sum = num1 ^ num2
        carry = (num1 & num2) << 1
        num1 = sum
        num2 = carry
    return num1

# print(two_num_sum(7, 6))

# 无重复字符的最长子串
def lengthOfLongestSubstring(s):
    if not s:
        return None
    res = ''
    max_len = 0
    for ch in s:
        if ch not in res:
            res += ch
            max_len = max(max_len, len(res))
        else:
            res += ch
            res = res[res.index(ch)+1:]
    return max_len

# print(lengthOfLongestSubstring('abcabcbb'))

# 寻找两个有序数组的中位数
# 给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
def findMedianSortedArrays(num1, num2):
    # 方法1：合并为1个大有序数组，然后查找中位数 O(m+n)
    # 方法2：针对有序/基本有序，优先考虑二分法
    pass


# 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
# 找出所有满足条件且不重复的三元组。
def three_sum(nums):
    size = len(nums)
    if not nums or size < 3:
        return []

    # 排序+ 双指针
    res = []
    nums.sort()
    for i in range(size):
        # 当前元素大于0,后面必然不会出现三元素和等于0,直接返回即可
        if nums[i] > 0:
            return res
        # 排序后相邻两数如果相等，则跳出当前循环继续下一次循环，相同的数只需要计算一次
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, size - 1
        while left < right:
            tmp = nums[i] + nums[left] + nums[right]
            if tmp == 0:
                res.append([nums[i], nums[left], nums[right]])
                # 重复元素跳过
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif tmp < 0:
                left += 1
            else:
                right -= 1
    return res

# print(three_sum([-1, 0, 1, 2, -1, -4]))


# 将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的
def merge_list(l1, l2):
    new_head = ListNode(0)
    cur_node = new_head
    while l1 or l2:
        if l1.val < l2.val:
            cur_node.next = l1
            l1 = l1.next
        else:
            cur_node.next = l2
            l2 = l2.next
    cur_node.next = l1 or l2
    return new_head.next


# 旋转数组中查找指定数字
def find_num_in_rotate_array(nums, target):
    if not nums:
        return -1
    # 二分法
    low, high = 0, len(nums)-1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < nums[high]:  # mid执向的数字在后面的有序子数组中
            if nums[mid] < target <= nums[high]:  # 右边查找
                low = mid + 1
            else:
                high = mid - 1
        else:  # 左半边有序
            if nums[low] <= target < nums[mid]:  # 左边查找
                high = mid - 1
            else:
                low = mid + 1