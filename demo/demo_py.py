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
        cur_node = cur_node.next
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


# 两数之和
# 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标
def sum_two_num(nums, target):
    res = []
    if not nums:
        return res

    # 方法1：双重循环, O(N*N)
    def sum_two_num_helper():
        _len = len(nums)
        for i in range(_len):
            for j in range(i, _len):
                if nums[i] + nums[j] == target and nums[i] != nums[j]:
                    res.extend([i, j])
                    break
    # sum_two_num_helper()

    # 方法2：哈希表，O(N)
    def sum_two_num_helper_hash():
        hash_table = {}
        for ix, num in enumerate(nums):
            hash_table[ix] = num
        for ix, num in hash_table.items():
            ret = target - num
            if ret in nums:
                if nums.index(ret) != ix:  # 非同一个数字
                    res.extend([ix, nums.index(ret)])
                    break
    # sum_two_num_helper_hash()

    def sum_two_num_helper_hash_once():
        # 遍历一次哈希标，在构建的同时检查
        hash_table = {}
        for ix, num in enumerate(nums):
            ret = target - num
            if ret in hash_table.values():
                ret_ix = nums.index(ret)
                if ret_ix != ix:
                    res.extend([ix, ret_ix])
                    break
            else:
                hash_table[ix] = num
    sum_two_num_helper_hash_once()
    return res

# print(sum_two_num([2, 7, 11, 15], 9))


# 两数相加
# 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
# 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
# 输出：7 -> 0 -> 8
# 原因：342 + 465 = 807
def two_sum_link(head1, head2):
    if not head1 or not head2:
        return head1 or head2
    carry = 0
    new_head = ListNode(-1)
    cur_node = new_head
    while head1 or head2:
        x = head1.val if head1 else 0
        y = head2.val if head2 else 0
        _sum = x + y + carry

        node = ListNode(_sum % 10)
        cur_node.next = node
        cur_node = cur_node.next
        carry = _sum / 10

        if head1:
            head1 = head1.next
        if head2:
            head2 = head2.next

    if carry > 0:  # 最高位进位
        node = ListNode(carry)
        cur_node.next = node
    return new_head.next


# 两个数相加，不用加法实现
# 不考虑进位: 0+0=0, 1+1=0, 0+1=1，等于异或操作
# 考虑进位: 0+1=0,1+1=0，有进位，等于与操作后左移一位
# 终止条件: 没有进位为止
def two_sum_without_add(num1, num2):
    while num2 > 0:
        _sum = num1 ^ num2
        carry = (num1 & num2) << 1
        num1 = _sum
        num2 = carry
    return _sum
# print(two_sum_without_add(5, 7))


# 无重复字符的最长子串
# 思路：sub_str记录最长子串，max_len记录长度
def maxlenOfSubStr(s):
    if not s or len(s) == 1:
        return s
    sub_str = ""
    max_len = 0
    for ch in s:
        if ch not in sub_str:   # ch没有重复，则更新max_len
            sub_str += ch
            max_len = max(max_len, len(sub_str))
        else:                   # ch有重复，则去掉前面的ch字符
            sub_str += ch
            sub_str = sub_str[sub_str.index(ch)+1:]
    return max_len, sub_str
# print(maxlenOfSubStr("abaa"))


# 寻找两个有序数组的中位数
def getMidTwoArray(arr1, arr2):
    # 方法1：构建一个大的有序数组后找中位数
    i = j = 0
    res = list()
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            res.append(arr1[i])
            i += 1
        else:
            res.append(arr2[j])
            j += 1
    res.extend(arr1[i:])
    res.extend(arr2[j:])
    return res[len(res) // 2]
# print(getMidTwoArray([1,2,3], [4,5,6]))


# 盛最多水的容器
# 直的两条线段将会与坐标轴构成一个矩形区域，较短线段的长度将会作为矩形区域的宽度，两线间距将会作为矩形区域的长度，
# 而我们必须最大化该矩形区域的面积。
def maxArea(heights):
    # 双指针法
    if not heights:
        return 0
    i, j = 0, len(heights)-1
    max_area = 0
    while i != j:
        area = min(heights[i], heights[j]) * (j - i)
        if area > max_area:
            max_area = area
        # 移动指针规则：移动小的
        if heights[i] < heights[j]:
            i += 1
        else:
            j -= 1
    return max_area
# print(maxArea([1,8,6,2,5,4,8,3,7]))


# 三数之和
# 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
def three_num_sum(nums):
    res = []
    if not nums or len(nums) < 3:
        return res
    nums.sort()
    size = len(nums)
    for i in range(size):
        if nums[i] > 0:  # 当前元素大于0，则后面的也必然大于0
            return res
        if i > 0 and nums[i] == nums[i-1]:  # 重复元素跳过
            continue
        left = i + 1
        right = size - 1
        while left < right:
            tmp = nums[i] + nums[left] + nums[right]
            if tmp == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif tmp < 0:
                left += 1
            else:
                right -= 1
    return res
# print(three_num_sum([-1, 0, 1, 2, -1, -4]))


# 合并两个有序的链表
# 新链表是通过拼接给定的两个链表的所有节点组成的
def mergeTwoLinkList(head1, head2):
    if not head1 or not head2:
        return head1 or head2
    head = tmp = ListNode(0)
    while head1 and head2:
        if head1.val <= head2.val:
            tmp.next = head1
            head1 = head1.next
        else:
            tmp.next = head2
            head2 = head2.next
        tmp = tmp.next

    tmp.next = head1 or head2
    return head.next


# 搜索旋转排序数组，[0,1,2,4,5,6,7] -> [4,5,6,7,0,1,2]
# 思路：有序(或基本有序)数组中查找某元素，使用二分法
def search(nums, target):
    if not nums:
        return -1
    start, end = 0, len(nums)-1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > nums[start]:  # mid元素大于start元素，则mid元素属于第一个有序子数组
            if nums[start] <= target < nums[mid]:  # 在左半子数组的左边查找
                end = mid - 1
            else:
                start = mid + 1
        else:
            if nums[mid] < target <= nums[end]:
                start = mid + 1
            else:
                end = mid - 1
    return -1
# print(search([4,5,6,7,0,1,2], 0))


# 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置
# 思路: 本题属于二分查找的变形，需要找到第一个和最后一个
def findFirstAndLastNum(nums, target):
    res = []
    if not nums:
        return res
    low, high = 0, len(nums)-1
    while low <= high:
        mid = low + (high - low) // 2
        if nums[mid] == target:
            first = last = mid
            while first >= 0 and nums[first-1] == target:  # 左边是target
                first -= 1
            while last <= len(nums)-1 and nums[last+1] == target:  # 右边是target
                last += 1
            return [first, last]
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return res
# print(findFirstAndLastNum([1,2,2,2,3,4,5], 2))


# 组合总数
# 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合
def combineSum(nums, target):
    if not nums:
        return nums

# 最大子序列和
def maxSubSum(nums):
    if not nums:
        return nums
    cur_sum = max_sum = float("-inf")
    for num in nums:
        if cur_sum <= 0:
            cur_sum = num
        else:
            cur_sum += num
        if cur_sum > max_sum:
            max_sum = cur_sum
    return max_sum
# print(maxSubSum([-2,1,-3,4,-1,2,1,-5,4]))


# 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
# 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
def clipFloor(n):
    # 思路：动态规划，dp[i]表示i阶台阶的方法，dp[i]=dp[i-1]+dp[i-2]
    # 初始: dp[0]=0,dp[1]=1,dp[2]=2
    dp = [None] * (n+1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
# print(clipFloor(4))


# 子集：给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
def getSub(nums):
    # 思路：初始子集只有[], 从前往后遍历，遇到一个数就把所有子集加上该数组成新的子集，遍历完毕即是所有子集
    # [] -> [],[1] -> [],[1],[2],[1,2] -> [],[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]
    res = []
    if not nums:
        return res
    res.append([])
    for num in nums:
        new = [tmp+[num] for tmp in res]
        res = res + new
    return res
# print(getSub([1, 2, 3]))


# 对称的二叉树
# 给定一个二叉树，检查它是否是镜像对称的
def isMirror(root):
    # 空树认为是对称的
    if not root:
        return True
    return isMirrorHelper(root.left, root.right)


def isMirrorHelper(left, right):
    if left is None and right is None:  # 两者都是空，表示已经遍历完成
        return True
    if left is None or right is None:   # 有一个为空，一个非空，不对称
        return False
    if left.data != right.data:         # 值不同，不对称
        return False
    return isMirrorHelper(left.left, right.right) and isMirrorHelper(left.right, right.left)


# 二叉树的最大深度
# 树的深度 = max(左子树的深度，右子树的深度) + 1
def maxDeepTree(root):
    if root is None:
        return 0
    left = maxDeepTree(root.left)
    right = maxDeepTree(root.right)
    return max(left, right) + 1


# 路径和：判断是否存在路径，使得路径值和等于target
def checkHasPath(root, target):
    # 先序遍历
    if root is None:
        return False

    # 到达叶子节点，且剩余值等于节点值
    if root.left is None and root.right is None and target == root.val:
        return True
    # 递归遍历左右子树
    return checkHasPath(root.left, target-root.val) or checkHasPath(root.right, target-root.val)


# 路径总和，找出所有路径
def getPath(root, target):
    res = []
    if root is None:
        return res

    def helper(node, tmp, sum):
        if node is None:
            return
        if node.left is None and node.right is None and sum == node.val:
            tmp.append(node.val)
            res.append(tmp)
        helper(node.left, tmp+[node.val], sum-node.val)
        helper(node.right, tmp+[node.val], sum-node.val)

    helper(root, [], target)
    return res


# 最小编辑距离
def minEditDist(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0 for i in range(n+1)] for i in range(m+1)]

    # base case
    for i in range(1, m+1):  # 表示s2为空，则最小距离为s1的字符个数
        dp[i][0] = i
    for j in range(1, n+1):  # 表示s1为空，则最小距离为s2的字符个数
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:  # 下标-1
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

print(minEditDist('batyu', 'beatyu'))


# 只出现1次的元素: 其余数字出现2次，只有1个数字出现1次
def findNumOnceApperance(nums):
    # 方法1：哈希表 时间复杂度O(N)  空间复杂度O(N)
    # 异或：相同为0 不同为1 0^x = x
    res = 0
    for num in nums:
        res ^= num
    return res
# print(findNumOnceApperance([2,1,2,1,3,4,4]))

# 环形链表：判断链表中是否有环
def hasCircle(head):
    # 思路：快慢指针，快指针每次走2步，慢指针每次走1步，
    if head is None and head.next is None:
        return False
    slow = head
    fast = slow.next
    # 终止条件: 已经是空或者是最后一个节点
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


# 反转链表
def reverseLinkList(head):
    # 方法1：三指针法
    if not head:
        return head

    def three_point():
        pPre = None
        pCur = head
        while pCur:  # 终止条件: 当前指针为空，表示全部都已经反转了
            tmp = pCur.next
            pCur.next = pPre
            pPre = pCur
            pCur = tmp
        return pPre

    def reverseHelper(pPre, pCur):
        # 尾递归
        if pCur:  # 当前为空，则返回前一个节点
            return pPre
        tmp = pCur.next
        pCur.next = pPre
        reverseHelper(pCur, tmp)
    reverseHelper(None, head)


# 反转二叉树
# 左右反转，即节点的左孩子等于节点的右孩子，节点的右孩子等于节点的左孩子
def reverseTree(root):
    # 线序遍历
    if root is None:
        return root
    root.left, root.right = root.right, root.left  # 交换左右子节点
    reverseTree(root.left)
    reverseTree(root.right)
    return root


# 回文数：正读和反读都相等
def isPalindromeNum(num):
    s = str(num)
    return s == s[::-1]

def isPalindromeStr(s):
    # 回文串
    new_str = filter(str.isalnum(), s.lower())  # 大小写不敏感，排除非字母或数字
    return new_str == new_str[::-1]

def isPalindromeLink(head):
    # 回文列表
    # 历列表把元素全部复制到list中，判断list是否回文
    nums = []
    while head:
        nums.append(head.val)
        head = head.next
    return nums == nums[::-1]


# 移动零
# 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序
def removeZero(nums):
    if not nums:
        return nums
    # 方法1: 利用2个辅助list, 时间复杂度O(N), 空间复杂度O(N)
    def helper():
        l1, l2 = [], []
        for num in nums:
            if num == 0:
                l1.append(num)
            else:
                l2.append(num)
        return l2 + l1

    def helper_point():
        # 原址排序，双指针法
        # 定义两个指针i,j，然后遍历数组，i跟j同时往前走，当遇到0时j停下，i继续往前走。当nums[i]不为0时则将num[i]的元素赋给j的位置，j++,nums[i]被赋值为0
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:  # 把i不为0的放到前面，j来指定放置的位置
                nums[j] = nums[i]
                if i != j:  # i!=j，表示当前有0，则把i位置置为0
                    nums[i] = 0
                j += 1
    helper_point()
nums = [0,1,2,0,3]
# removeZero(nums)
# print(nums)


# 找到所有数组中消失的数字
def findDisappearedNumbers(nums, n):
    if not nums:
        return []
    return set(range(1, n+1)) - set(nums)
# print(findDisappearedNumbers([1,3,3,5,5], 5))


# 把二叉搜索树转换为累加树
pre_node_val = 0
def bstToConvertTree(root):
    # 思路: 二叉搜索树性质 - 左节点值 < 根节点值 < 右节点值
    #   因此可以从右-根-左的顺序来遍历，把当前节点值加上前一个节点值
    if not root:
        return None
    bstToConvertTree(root.right)

    global pre_node_val
    root.val += pre_node_val
    pre_node_val = root.val

    bstToConvertTree(root.left)
    return root


# 合并二叉树
def mergeTree(root1, root2):
    # 先序遍历
    # 在遍历时，如果两棵树的当前节点均不为空，我们就将它们的值进行相加，并对它们的左孩子和右孩子进行递归合并；
    # 如果其中有一棵树为空，那么我们返回另一颗树作为结果；如果两棵树均为空，此时返回任意一棵树均可（因为都是空）
    if not root1 or not root2:
        return root1 or root2
    # 合并节点
    root1.val = root1.val + root2.val
    root1.left = mergeTree(root1.left, root2.left)
    root1.right = mergeTree(root1.right, root2.right)
    return root1


# 数字二进制中1的个数
def countOfOne(num):
    cnt = 0
    while num:
        num = num & (num-1)
        cnt += 1
    return cnt
# print(countOfOne(12))


# 汉明距离：两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目
def hammingDistance(num1, num2):
    # 方法: 本体要找二进制不同的数目，异或：相同为0，不同为1
    # 异或后问题可转换为求二进制中1的个数
    num = num1 ^ num2
    return countOfOne(num)
# print(hammingDistance(12, 7))


# 最小栈
class MinValueStack:
    def __init__(self):
        self.stack = []
        self.min_value = []

    def push(self, item):
        self.stack.append(item)
        if not self.min_value:
            self.min_value.append(item)
        elif self.min_value[-1] < item:
            self.min_value.append(self.min_value[-1])
        else:
            self.min_value.append(item)

    def pop(self):
        if self.stack:
            self.stack.pop()
            self.min_value.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]

    def get_min(self):
        if self.min_value:
            return self.min_value[-1]


# 多数元素：次数超过一般的元素
def moreHalfNum(nums):
    # 方法1：排序后找中位数
    # 方法2：相同元素次数加1，不同元素次数-1，最后次数为正的数字即为所求
    time = 1
    num = nums[1]
    for i in range(1, len(nums)):
        if nums[i] == num:
            time += 1
        else:
            time -= 1
        if time <= 0:  # 次数小于等于0，则更新num
            num = nums[i]
    return num
# print(moreHalfNum([1,2,2,2,2,1,1,1,2,2]))


# 相交链表：求两个链表的第一个公共节点
# 快慢指针法
def findFirstCommonNode(head1, head2):
    if not head1 or not head2:
        return None
    # step1:计算两个链表的长度
    len_head1 = len_head2 = 0
    node1, node2 = head1, head2
    while node1:
        len_head1 += 1
        node1 = node1.next
    while node2:
        len_head2 += 1
        node2 = node2.next

    # step2: 长的链表先走abs(len_head1 - len_head2)步
    node1, node2 = head1, head2
    if len_head1 > len_head2:
        for i in range(len_head1 - len_head2):
            node1 = node1.next
    else:
        for i in range(len_head2 - len_head1):
            node2 = node2.next

    # step3：两个链表一起都，直到相遇或者走完为止
    while node1 and node2:
        if node1 == node2:
            return node1
        node1 = node1.next
        node2 = node2.next
    return None


# 合并两个有序数组，要求在num1上合并
# 方法1：使用辅助数组，一次遍历两个数组 缺点：需要额外空间O(m+n)
# 方法2：直接列表相加后，排序 O(N*logN)
# 方法2：正常情况一般从头开始比较，但是这样会导致num1数组移位，最坏情况O(N*N), 转换思维，从尾开始比较
def mergeNums(num1, m, num2, n):
    if not num1 or len(num1) < m or not num2 or len(num2) < n or len(num1) + len(num2) < m+n:
        return None

    # 比较num1，num2，直接计算应该插入的位置
    while m > 0 and n > 0:
        if num1[m-1] <= num2[n-1]:
            num1[m+n-1] = num1[m-1]
            m -= 1
        else:
            num1[m+n-1] = num2[n-1]
            n -= 1
    # 处理nums2还有元素,表示nums2的元素比较小，应该在最前面
    if n > 0:
        num1[:n] = num2[:n]

# 两个数组的交集
def intersection(num1, num2):
    # 要求：输出结果中的每个元素一定是唯一的
    # return set(num1) & set(num2)

    # 要求：输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致
    dict_num = {}
    for num in num1:
        dict_num[num] = dict_num[num] + 1 if num in dict_num else 1

    ans = []
    for num in num2:
        if num in dict_num:
            ans.append(num)
            dict_num[num] -= 1
    return ans
# print(intersection([4,9,5], [1,2,9,4,1,2]))

# 两整数之和
def sumTwoNum(a, b):
    sum = carry = 0
    while b:
        sum = a ^ b
        carry = (a & b) << 1
        a = sum
        b = carry
    return a
# print(sumTwoNum(7, 6))


# 字符串中第一个唯一字符
# 给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1
def findFirstOnceStr(s):
    if not s:
        return -1
    # 用哈希表来计算出现次数
    hash_table = {}
    for ch in s:
        if ch in hash_table:
            hash_table[ch] += 1
        else:
            hash_table[ch] = 1
    for i, ch in enumerate(s):
        if hash_table[ch] == 1:
            return i
    return -1

    # 方法2:利用python的count()
    # for i, ch in enumerate(s):
    #     if s.count(ch) == 1:
    #         return i
# print(findFirstOnceStr("leetcode"))


# 左叶子之和
# 计算给定二叉树的所有左叶子之和(同理可引申到右叶子之和)
# 难点:要判断该叶子节点是左叶子
# 叶子: node.left is None and node.right is None
# 左叶子: node.left is not None and node.left.left is None and node.left.right is None -> 说明node.left是node的左叶子
def leftNodeSum(root):
    if not root:
        return 0
    # 先序遍历：从根节点出发
    # 叶子节点
    if root.left is not None and root.left.left is None and root.left.right is None:
        return root.left.val + leftNodeSum(root.right)
    else:
        return leftNodeSum(root.left) + leftNodeSum(root.right)

# 题目描述：给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数
def twoSum(nums, target):
    # 方法：first = 0， end = len(nums)-1
    #   1.nums[first] + nums[end] > target: end - 1
    #   2.nums[first] + nums[end] > target: first + 1
    #   3.nums[first] + nums[end] == target: 返回
    if not nums:
        return []
    first, end = 0, len(nums)-1
    while first < end:
        if nums[first] + nums[end] > target:
            end -= 1
        elif nums[first] + nums[end] < target:
            first += 1
        else:
            return [first, end]
    return None
# print(twoSum([2,7,9,11], 9))


# 删除排序数据中的重复项
def deleteDuplicateNums(nums):
    if not nums:
        return None

    def helper1():
        # 方法1: 遍历数组，当前i和下一个相等，则删除下一个，直到不相等位置
        for i in range(len(nums)):  # TODO:这里必须在迭代中显示len(nums),因为有删除动作会影响nums的长度
            if i < len(nums)-1 and nums[i] == nums[i+1]:
                nums.pop(i+1)
        return len(nums)
    helper1()

# nums = [1,2,2,3,3,4]
# deleteDuplicateNums(nums)
# print(nums)


# 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一
def addOne(nums):
    if not nums:
        return []
    # 先转换为整数，加1后在转换为列表
    res = 0
    for num in nums:
        res = res * 10 + num

    res += 1
    ret = []
    while res:
        ret.insert(0, res % 10)
        res //= 10
    return ret
# print(addOne([1,2,3]))


# 反转整数
def reverseNum(num):
    # 方法1：常规做法，利用num/pop, num/=10,组个为新的数字
    def helper(num):
        nega_flag = False if num > 0 else True
        num = abs(num)
        res = 0
        while num:
            remainder = num % 10
            num //= 10
            res = res * 10 + remainder
        return 0-res if nega_flag else res
    # return helper(num)

    def helper2(num):
        # 利用python语法
        num_str = str(abs(num))[::-1]
        return int(num_str) if num > 0 else (0-int(num_str))
    return helper(num)
# print(reverseNum(-123))


# 两个队列实现栈
class Solution:
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def push(self, item):
        # 入栈：随便入一个队列即可
        self.queue1.append(item)

    def pop(self):
        # 出栈：后入先出
        # 出队列1元素插入到队列2，直到队列1中只有一个元素
        # 交换队列1，队列2；弹出队列2
        if len(self.queue1) == 0:
            return None
        while len(self.queue1) > 1:
            self.queue1.append(self.queue1.pop(0))

        self.queue1, self.queue2 = self.queue2, self.queue1
        return self.queue2.pop()


class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, item):
        self.stack1.append(item)

    def pop(self):
        if not self.stack1 and not self.stack2:
            raise Exception("stack is empty")

        if len(self.stack2) > 0:
            return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()


# 和最少为k的最短子数组
# 返回 A 的最短的非空连续子数组的长度，该子数组的和至少为 K 。
def findMinLen(nums, k):
    if not nums:
        return -1

    def helper1():
        # 思路: 遍历出所有和至少为k的子数组，然后找最短
        #   如何找出所有的子数组呢？从上一搜索位置的下一位置开始查找
        # O(N*N)
        res = []
        nums_len = len(nums)
        for i in range(nums_len):
            _sum = 0
            for j in range(i, nums_len):
                _sum += nums[j]
                if _sum >= k:
                    res.append(j - i + 1)  # 找到和大于k的子数组后记录长度，并跳出循环，因为继续往后找子数组长度只会更大
                    break
        return min(res)
# print(findMinLen([2,-1,2], 3))


# 设计循环队列
# 循环队列有头尾指针，主要在于要判断是否为空，是否满了
class Solution:
    def __init__(self, k):
        self.arr = [None] * (k+1)
        self.front = 0  # 头指针
        self.rear = 0   # 尾指针

    def is_empty(self):
        # 头尾指针相等，说明当前循环列表为空，并不一定都等于0
        return self.front == self.rear

    def is_full(self):
        # 头尾指针相差1，说明列表已满没有空余位置
        return (self.rear + 1) % len(self.arr) == self.front

# 删除链表的倒数第N个节点
# 给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点
def deleteNode(head, n):
    if not head or n < 0:
        return None

    def helper1():
        # 思路：首先遍历列表统计列表长度len；然后从开走len-n-1步，该节点是待删除节点的前一节点，删除即可
        _len = 0
        node = head
        while node:
            _len += 1
            node = node.next
        if n > _len:
            return None
        if n == _len:  # 表示删除第一个节点
            return head.next
        node = head
        for i in range(_len-n-1):
            node = node.head
        node.next = node.next.next
        return head

    def helper2():
        # 快慢指针法: 快指针先走n步，然后一起走，快指针走到末尾，满指针即为待删除节点的前一个节点
        fast = slow = head
        list_len = 0
        for i in range(n):
            if fast:
                fast = fast.next
                list_len = i
        if list_len < n:  # 链表长度小于n
            return None

        if fast is None:  # n等于链表长度
            return head.next

        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head


#A 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次
def dropDuplicateList(head):
    # 思路：2指针cur，last
    if not head or not head.next:
        return head
    cur = head
    last = head.next
    while cur:  # 终止条件: 遍历到最后一个节点
        if cur.val == last.val:  # 当前节点和下一个重复，则移动last
            while last and last.val == cur.val:
                last = last.next
            cur.next = last
            cur = last
            if last.next:
                last = last.next
    return head

# 环的入口节点
# 给定一个链表，返回链表开始入环的第一个节点
def detectCycle(head):
    if not head or not head.next:
        return -1

    # step1: 检查是否有环
    node = has_cycle(head)
    if not node:
        return -1

    # step2: 确认环中节点个数num
    node_nums = cycle_node_nums(node)

    # step3：快慢指针，快指针先走num步，然后一起走，第一次相遇的节点即为环的入口节点
    return first_detect_cycle(head, node_nums)


def has_cycle(head):
    # 检查是否有环：快慢指针法
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return fast
    return None

def cycle_node_nums(node):
    # 确定环中的节点数，快慢指针法
    slow = node
    cnt = 1
    while slow.next != node:
        slow = slow.next
        cnt += 1
    return cnt

def first_detect_cycle(head, cnt):
    # 寻找入口节点: 快慢指针
    fast = slow = head
    for i in range(cnt):
        fast = fast.next

    while fast != slow:
        fast = fast.next
        slow = slow.next
    return fast


# 编写一个程序，找到两个单链表相交的起始节点
def find_first_common_node(head1, head2):
    # 思路：计算两个链表的长度len1，len2，长的链表先走abs(len1-len2)步，然后一起走，第一次相遇的节点即为所求
    if not head1 or not head2:
        return None
    len1 = len2 = 0
    node = head1
    while node:
        len1 += 1
        node = node.next
    node = head2
    while node:
        len2 += 1
        node = node.next

    node1, node2 = head1, head2
    if len1 > len2:
        for i in range(len1 - len2):
            node1 = node1.next
    else:
        for i in range(len2 - len1):
            node2 = node2.next

    while node1 and node2:
        if node1 == node2:
            return node1
        node1 = node1.next
        node2 = node2.next
    return None


# 移除链表元素：删除链表中等于给定值 val 的所有节点
def removeNode(head, num):
    if not head or head.val == num:
        return None
    # node = head
    # while node and node.next:
    #     if node.next.val == num:
    #         node.next = node.next.next
    #         node = node.next
    # return head
    # 双指针
    firstnode = ListNode(0)
    firstnode.next = head
    pre, cur = firstnode, head
    while cur:
        if cur.val == num:
            pre.next = cur.next
            cur = cur.next
        else:
            pre = pre.next
            cur = cur.next
    return firstnode.next


# 反转列表
def reverseLinkList(head):

    if not head or not head.next:
        return head

    def helper1():
        # 三指针法：把当前节点的next指向上一节点
        first_node = ListNode(0)
        first_node.next = head
        pre, cur = first_node, head
        while cur:
            last = cur.next
            cur.next = pre
            pre = cur
            cur = last
        return first_node.next

    def helper2(pre_node, cur_node):
        # 递归法
        if cur_node is None:
            return pre_node
        last_node = cur_node.next
        cur_node.next = pre_node
        return helper2(cur_node, last_node)
    return helper2(None, head)

# 回文链表
def isPalindromeLink(head):
    if not head or not head.next:
        return True
    # 方法：遍历链表，把链表元素存储到列表中，判断列表是否回文
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res == res[::-1]

# 相同的树
# 结构相同，值相同
def is_same_tree(root1, root2):
    if not root1 and not root2:   # 2个树都遍历完成
        return True
    if not root1 or not root2:    # 1个遍历完，1个没有
        return False
    if root1.val != root2.val:    # 值不相等
        return False
    # 递归遍历左右子树
    return is_same_tree(root1.left, root2.left) and is_same_tree(root1.right, root2.right)


# 对称二叉树
def isSymmetric(root):
    if root is None:
        return True
    return isSymmetric_helper(root.left, root.right)

def isSymmetric_helper(left_root, right_root):
    if not left_root and not right_root:
        return True
    if not left_root or not right_root:
        return False
    if left_root.val != right_root.val:
        return False
    # 递归左子树的左孩子和右子树的右孩子 and 左子树的右孩子和右子树的左孩子
    return isSymmetric_helper(left_root.left, right_root.right) and isSymmetric_helper(left_root.right, right_root.left)

# 二叉树的最大深度
def max_depth(root):
    if not root:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return max(left_depth, right_depth) + 1


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

# 将有序数组转换为二叉搜索树
# 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树
# 方法: BST树的中序遍历是有序的，因此可以把有序数组看作是BST的中序遍历，关键在于如何构建树
#   1、构建根节点：在数组中查找key，key前面的数小于key，后面的数大于key
#   2、递归构建左右子树
#   3、平衡树：左右子树的高度差不超过1，因此可以直接取中间值作为根节点(特殊性质)
def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    return root

# 二叉树的最小深度
# 最小深度是从根节点到最近叶子节点的最短路径上的节点数量
def root_min_depth(root):
    if not root:
        return 0
    def helper1():
        # 层次遍历：遇到叶子节点就返回最小深度
        depth = 1
        queue = [root]
        while queue:
            tmp = []
            for node in queue:
                if node.left is None and node.right is None:
                    return depth
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            queue = tmp
            depth += 1
        return depth


# 平衡二叉树
# 左右子树的高度相差不能大于1
def isBalanceTree(root):
    # 先序遍历，计算左右子节点的深度，判断是否平衡
    # 缺点：从上到下，计算深度的时候会有很多重复计算
    if not root:
        return True

    # 判断节点的左右子树是否平衡
    depth_diff = max_depth(root.left) - max_depth(root.right)
    if abs(depth_diff) > 1:
        return False
    # 递归遍历左右子树
    return isBalanceTree(root.left) and isBalanceTree(root.right)


def isBalanceTree_v2(root):
    # 考虑从下往上遍历，若某个子树不平衡，则整棵树必然不平衡
    # 后续遍历实现从下往上
    if not root:
        return True

    flag = True  # 标记是否是平衡树
    def treeDepth(root):
        if not root:
            return 0
        left = treeDepth(root.left)
        right = treeDepth(root.right)
        if abs(left - right) > 1:
            global flag
            flag = False
        return max(left, right) + 1

    treeDepth(root)
    return flag

# 最长公共前缀
def maxLenCommonPrefix(strs):
    # 思路: 找出列表中最短的字符串，然后判断公共前缀
    if not strs:
        return None
    min_str = min(strs, key=len)  # 选择长度最短的字符串
    for i, ch in enumerate(min_str):
        for other in strs:
            if other[i] != ch:  # 第i个字符不等，说明ch非公共前缀
                return min_str[:i]
    return min_str
# print(maxLenCommonPrefix(['abc', 'accd', 'ab']))


def strStr(haystack, needle):
    # 字符串匹配，haystack: 模式串，needle: 匹配串
    # 方法1：
    # return haystack.index(needle)

    # 方法2
    def strStr_v2(haystack, needle):
        i = j = 0
        while i < len(haystack) and j < len(needle):
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                i = i - j + 1  # 不匹配，则返回本次匹配的下一位置
                j = 0
        if j == len(needle):  # 完全匹配，返回needle在haystack中的起始位置
            return i - j
        return -1
    return strStr_v2(haystack, needle)
# print(strStr("abcde", "ce"))

# 二进制求和
# 方法1：类似于两数相加，异或和与运算
# 方法2：python语法，先转为10进制，相加后在转为二进制

# 反转字符串
# return str[::-1]

# 滑动窗口最大值
import collections
def maxSlidingWindow(nums, k):
    res = []
    if not nums:
        return res
    n = len(nums)
    deque = collections.deque()
    # i: [1-k, n-k+1]   j: [0: n-1]
    for i, j in zip(range(1-k, n-k+1), range(0, n)):
        # 始终让最大值位于双端队列的头部
        # 1、如果出窗口的元素是当前最大元素，则删除最大元素
        if i > 0 and nums[i-1] == deque[0]:
            deque.popleft()
        # 2、如果当前队列中的值小于Num[j],则删除这部分
        while deque and deque[-1] < nums[j]:
            deque.pop()
        deque.append(nums[j])
        if i >= 0:  # 说明有了完整的窗口
            res.append(deque[0])
    return res
# print(maxSlidingWindow([5,3,4], 2))

class MaxQueue(object):
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def push_bask(self, item):
        self.queue1.append(item)
        # 始终把最大值放到辅助队列开头
        while self.queue2 and self.queue2[-1] < item:
            self.queue2.pop()
        self.queue2.append(item)

    def pop_front(self):
        if not self.queue1:
            return -1
        ans = self.queue1.pop(0)
        if ans == self.queue2[0]:  # 删除的最大值
            self.queue2.pop(0)
        return ans

    def max_value(self):
        if not self.queue2:
            return -1
        return self.queue2[0]
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import GradientBoostingRegressor


# 扑克牌中的顺子
def isStraight(nums):
    # 方法1：排序后记录大小王个数和相邻牌的间隔，大小王数 > 相邻间隔 则为顺子
    # 方法2：不考虑大小王，若为顺子，则必然最大牌 - 最小牌 < 5
    if not nums:
        return False
    card = {}
    for num in nums:
        if 0 == num:  # 大小王不参与判断
            continue
        if num in card:
            return False
        card[num] = 0
    return True if max(card.values()) - min(card.values()) < 5 else False
# print(isStraight([1,1,2,3,5]))


# 在未排序的数组中找到第2大的数字
def findSecondMaxValue(nums, k):
    # 方法1：利用快排的Partition()，把数据排序，返回选中的ix，若ix=len(nums)-2即为第2大
    # 时间复杂度O(NlogN)  好处是可以扩展到查找任意大的数字
    if not nums or len(nums) < 2:
        return None
    def helper():
        length = len(nums) - 1
        index = partition(nums, 0, length)
        while index != k - 1:
            print(index)
            if index < k - 1:
                index = partition(nums, index + 1, length)
            if index > k - 1:
                index = partition(nums, 0, index - 1)
        return nums[index]
    # return helper()

    def helper2():
        # 方法2: 遍历数组，标记最大值和第2大值
        # 若num大于最大值，则把最大值赋值给次大值，更新最大值
        # 若num大于次大值，则更新次大值
        maxValue = secondMaxValue = float('-inf')
        for num in nums:
            print('maxValue:%s, secondMaxValue:%s, num:%s' % (maxValue, secondMaxValue, num))
            if num > maxValue:
                secondMaxValue = maxValue
                maxValue = num
            elif num > secondMaxValue:
                secondMaxValue = num
        return maxValue, secondMaxValue
    return helper2()

def partition(nums, start, end):
    pivot = nums[start]
    while start < end:
        while start < end and nums[end] >= pivot:
            end -= 1
        nums[start], nums[end] = nums[end], nums[start]
        while start < end and nums[start] <= pivot:
            start += 1
        nums[start], nums[end] = nums[end], nums[start]
    return start

# print(findSecondMaxValue([4, 3, 9, 8, 7, 20], 5))


# 最大子序列和
def maxSunArray(nums):
    if not nums:
        return None

    def helper1():
        # 方法1，遍历数据，记录当前和，最大和
        cur_sum = max_sum = float('-inf')
        for num in nums:
            if cur_sum <= 0:
                cur_sum = num
            else:
                cur_sum += num
            if cur_sum > max_sum:
                max_sum = cur_sum
        return max_sum
    # return helper1()

    def helper2():
        # 贪心算法，每次总使做出在当前看来最好的方案
        cur_sum = max_sum = float('-inf')
        for num in nums:
            cur_sum = max(cur_sum, cur_sum+num)
            max_sum = max(cur_sum, max_sum)
        return max_sum
    return helper2()

    def helper3():
        # 动态规划
        # 状态转移方程 dp[i] = num[i]+dp[i-1] if dp[i-1]>0; 否则 dp[i] = num[i]
        for i in range(len(nums)):
            nums[i] += nums[i-1] if nums[i-1] > 0 else 0
        return nums[-1]
    return helper3()
# print(maxSunArray([-2,1,-3,4,-1,2,1,-5,4]))


# 爬楼梯：n台阶楼梯，每次可以走1阶，可以走2阶，那么n阶共有多少种方法
def clipFloor(n):
    # 方法1：斐波那契法
    # f(0)=0,f(1)=1,f(2)=2,f(3)=3,...
    def clipFloor_fib():
        a, b = 1, 2
        for i in range(3, n+1):
            a, b = b, a + b
        return b
    # return clipFloor_fib()

    def clipFloor_dp():
        # 第i阶台阶的方法dp[i] = dp[i-1] + dp[i-2]
        dp = [None] * (n + 1)
        dp[1], dp[2] = 1, 2
        for i in range(3, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    return clipFloor_dp()
# 实际上动态规划的简易版就是方法1
# print(clipFloor(4))

# 区域和检索：给定一个整数数组  nums，求出数组从索引 i 到 j  (i ≤ j) 范围内元素的总和，包含 i,  j 两点。
def sumRange(nums, i, j):
    if not nums or i < 0 or j < 0 or i > j:
        return None
    # 方法1
    return sum(nums[i:j+1])
# print(sumRange([-3, 0, 2, 1, 6], 0, 2))

# 判断子序列：给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
# 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
def isSubStr(s, t):
    # 注意这里和常见的字符串比较不同
    def helper():  # O(N)
        p1 = p2 = 0
        while p1 < len(s) and p2 < len(t):
            if p1 == p2:
                p1 += 1
                p2 += 1
            else:
                p2 += 1  # 不匹配时只移动主串，子串不移动
        return p2 == len(s)
    return helper()
# print(isSubStr("ace", "abcde"))

# 字符串转整数
# 异常情况: 开头的首位正负号，非数字字符不影响转换
# 中间的非数字字符异常
def str2int(s):
    if not s:
        return None
    s = s.strip()
    num = ''
    for i in range(len(s)):
        if i == 0 and i not in ['-', '+'] and not s[i].isdigit():
            return None
        if i != 0 and not s[i].isdigit():
            return None
        num += s[i]
    return int(num)

# 字符串相加
def addStr(num1, num2):
    if not num1 or not num2:
        return num1 or num2
    sum1 = sum2 = 0
    for ch in num1:
        sum1 = sum1 * 10 + ord(ch) - ord('0')
    for ch in num2:
        sum2 = sum2 * 10 + ord(ch) - ord('0')
    return str(sum1 + sum2)
# print(addStr("123", "111"))

# 字符串中的单词数
def wordNumsInStr(s):
    if not s:
        return 0
    return len(s.strip().split())
# print(wordNumsInStr(" a b   c "))

# 压缩字符串

# 重复叠加字符串匹配
# 给定两个字符串 A 和 B, 寻找重复叠加字符串A的最小次数，使得字符串B成为叠加后的字符串A的子串，如果不存在则返回 -1
def repeatedStringMatch(A, B):
    # 如果A串长度比B串长度小，则A串加长
    cnt = 1
    s = A
    while len(s) < len(B):
        s += A
        cnt += 1
    if B in A:  # 判断B是否是A的子串
        return cnt
    s += A  # 如果不是可能是因为首尾没有拼接到一起，在加A即可拼接
    cnt += 1
    if B in s:
        return cnt
    return -1
# print(repeatedStringMatch(A = "abcd", B = "cdabcdab"))

# 最常见的单词
from collections import Counter
def mostCommonWord(paragraph, banned):
    word_list = paragraph.lower().split(',')
    word_count = Counter(word_list)
    res = word_count.most_common(len(banned)+1)  # 选择次数最多的X个单词
    for word_cnt in res:
        if word_cnt[0] not in res:
            return word_cnt[0]

# 按奇偶排序数组：奇数在偶数前面
def sortArrayByParity(nums):
    # 方法1：利用辅助列表，分别保存奇数和偶数，然后整合为一个大列表
    # if not nums:
    #     return []
    # res1, res2 = [], []
    # for num in nums:
    #     if num % 2 == 1:
    #         res1.append(num)
    #     else:
    #         res2.append(num)
    # return res2+res1

    return sorted(nums, key=lambda x: x % 2)  # 偶数余2为0，排在前面
# print(sortArrayByParity([3,1,2,4]))

# leetcode494 目标和
# 方法1：回溯法

class Test:
    def __init__(self):
        self.__nums = 0  # 私有变量，只能被类方法调用
        self._nums = 0   # protected变量，只能被类方法，类对象，子类调用

    def __get_num(self):  # 私有方法，只能被类方法调用,在类中被命名_Test__get_num
        return self.__nums

# obj = Test()
# print(obj._nums)
# print(obj._Test__get_num())

# 动态语言添加属性和方法
class Person:
    def __init__(self, name):
        self.name = name
#
# li = Person("李")
# li.age = 20   # 再程序没有停止下，将实例属性age传入。动态语言的特点
# Person.age = None  # 这里使用类名来创建一个属性age给类，默认值是None。Python支持的动态属性添加。
# def eat(self):
#     print("%s is eating" % self.name)
# import types
# # 使用types.MethodType，将函数名和实例对象传入，进行方法绑定。并且将结果返回给li.eat变量。
# # 实则是使用一个和li.eat方法一样的变量名用来调用。
# li.eat = types.MethodType(eat, li)
# print(li.eat())


# 两颗树最低公共祖先
# 当树为二叉搜索树，查找最低公共祖先节点的方法:
#   1、从根节点出发 -- 先序遍历
#   2、如果根节点值大于给定2个节点的值，则最低公共祖先必然在根节点的左子树
#   3、如果根节点值小于给定2个节点的值，则最低公共祖先必然在根节点的右子树
#   4、先序遍历，直到遇到某个节点值在给定2个节点值的中间，那么这个节点必然是最低公共祖先节点
def findCommonNode(root, node1, node2):
    # 树为二叉搜索树
    if not root or not node1 or not node2:
        return None
    node = findCommonNodeHelper(root, node1, node2)
    return node

def findCommonNodeHelper(root, node1, node2):
    if not root:
        return
    if root.val < node1.val and root.val < node2.val:
        findCommonNodeHelper(root.right, node1, node2)
    elif root.val > node1.val and root.val > node2.val:
        findCommonNodeHelper(root.left, node1, node2)
    else:
        return root

# 普通树，但是有指向父节点的指针
# 可以转换为求两个链表的第一个公共节点

# 没有指向父节点的指针
# 需要前序遍历树，保存从根节点到指定节点的路径，最后在路径中找最低公共祖先节点
def getPath(root, targetNode, path):
    if path and path[-1] == targetNode:  # 找到目标节点
        return
    if root is None:
        return
    # 先序遍历路径
    path.append(root)
    getPath(root.left, targetNode, path)
    getPath(root.right, targetNode, path)
    if path and path[-1] != targetNode:  # 回溯
        path.pop()


def findCommonNodePath(root, node1, node2):
    if root is None or node1 is None or node2 is None:
        return None
    path1, path2 = [], []
    getPath(root, node1, path1)
    getPath(root, node2, path2)

    # 从前往后，在2个路径中查找最后的相等节点，即为最低公共祖先节点
    result = None
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            result = path1[i]
    return result

# 数组中重复的数
#
def findRepeatNumber(nums):
    if not nums:
        return None
    def helper1():
        # 方法1:遍历数组,count
        for num in nums:
            if nums.count(num) > 1:
                return num
        return None

    def helper2():
        # 方法2: 利用哈希表
        hash_table = {}
        for num in nums:
            if num not in hash_table:
                hash_table[num] = 1
            else:
                return num
        return None
    # return helper2()

    def helper3():
        # 方法3: 利用数字是0~n-1的特性,如果没有没有重复,则可以是i位置的元素等于nums[i]
        for i in range(len(nums)):
            if i != nums[i]:
                if nums[nums[i]] == nums[i]:  # 如果nums[i]位置上已经是匹配元素,表示重复
                    return nums[i]
                # 把i位置的元素num[i]当道nums[nums[i]]位置上
                tmp = nums[i]
                nums[i], nums[tmp] = nums[tmp], nums[i]
    # return helper3()
# print(findRepeatNumber([2,3,1,0,2,5,3]))

def replaceSpace(s):
    if not s:
        return ""
    # 方法1
    # return s.replace(' ', '%20')

    # 方法2: 字符串不能修改,只能先转换为list - 修改list - 转换为字符串
    s = list(s)
    for i in range(len(s)):
        if s[i] == ' ':
            s[i] = '%20'
    return ''.join(s)
# print(replaceSpace("We are happy."))

# 从尾到头打印链表
def printListFromTailToHead(head):
    if not head:
        return []
    if not head.next:
        return [head.val]

    # 方法1: 利用辅助栈
    # def helper1():
    #     stack = []
    #     node = head
    #     while node:
    #         stack.append(node.val)
    #     return stack[::-1]

    # 方法2: 递归法
    printListFromTailToHead(head.next) + [head.val]

# 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# # 例如，给出
# # 前序遍历 preorder = [3,9,20,15,7]
# # 中序遍历 inorder = [9,3,15,20,7]
def rebuildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    # 思路:
    # 首先构建根节点: 在前序遍历中第一个元素为根节点,在中序遍历中找左子树和右子树节点
    loc = inorder.index(preorder[0])
    root = TreeNode(preorder[0])
    # 注意前序和中序列表的范围
    root.left = rebuildTree(preorder[1: loc + 1], inorder[: loc])
    root.right = rebuildTree(preorder[loc + 1:], inorder[loc + 1:])
    return root

class TwoStackToQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, item):
        # 入队列: 插入栈1
        self.stack1.append(item)

    def pop(self):
        # 出队列: 若栈2有元素则出栈2;否则把栈1元素依次弹出,插入到栈2中,这样在栈2中的元素顺序是先入先出
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        if self.stack2:
            return self.stack2.pop()
        return -1


class TwoQueueToStack:
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def push(self, item):
        # 入栈: 插入到队列1
        self.queue1.append(item)

    def pop(self):
        # 出栈: 把队列1的元素依次按照先入先出的顺序弹出压入到栈2中,直到阵列1中只有1个元素为止;交换队列1,2;出队列2
        while len(self.queue1) > 1:
            self.queue2.append(self.queue1.pop(0))
        self.queue1, self.queue2 = self.queue2, self.queue1
        if self.queue2:
            return self.queue2.pop()
        return -1

# 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
# 输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  
def minArray(nums):
    if not nums:
        return None
    # # 方法1: 从头到尾遍历,找最小  O(N)
    # min_val = float('inf')
    # for num in nums:
    #     if num < min_val:
    #         min_val = num
    # return min_val

    # 方法2: 有序数组或递增数据找数,首选二分法
    # 这个要确认mid指向的数字是前子数组还是后子数组
    # 归属前子数组: nums[mid] > nums[0]
    # 归属猴子数组: nums[mid] < nums[-1]
    start, end = 0, len(nums)-1
    # 中间条件:
    # start指向第一个有序子数组的最后一位
    # end指向第二个有序子数组的第一位, 因此有star+1 = end
    while nums[start] >= nums[end]:
        # 当两个指针走到挨着的位置时，即right - left == 1,right就是最小数了
        if end - start == 1:
            mid = end
            break
        mid = start + ((end - start) >> 1)  # 注意要有括号,>>优先级较低
        if nums[start] == nums[end] == nums[mid]:
            # 特殊情况: 逐个比较
            min_val = nums[start]
            for num in nums[start: end + 1]:
                if num < min_val:
                    return num
                min_val = num
        if nums[mid] >= nums[start]:  # 左子数组, 令start指向mid位置,仍然为左子数组
            start = mid
        else:                         # 右子数组, 令end指向mid位置,仍然为右子数组
            end = mid
    return nums[mid]
# print(minArray([3,4,5,1,2]))

# 矩阵中的路径
# 回溯法
def hashPath(matrix, rows, columns, path):
    if not matrix:
        return False
    visited = [[0] * columns for i in range(rows)]
    # 首先要在s中查找path的起始位置
    for i in range(rows):
        for j in range(columns):
            if matrix[i*columns+j] == path[0]:  # 确定起始位置
                if hashPathHelper(matrix, rows, columns, path[1:], i, j, visited):
                    return True
    return False

def hashPathHelper(matrix, rows, cols, path, i, j, visited):
    if not path:  # 遍历路径完成
        return True
    visited[i][j] = 1
    # 向下搜索
    if i + 1 < rows and matrix[(i+1)*cols+j] == path[0] and visited[i+1][j] == 0:
        return hashPathHelper(matrix, rows, cols, path[1:], i+1, j, visited)
    # 向上搜索
    elif i - 1 >= 0 and matrix[(i-1)*cols+j] == path[0] and visited[i-1][j] == 0:
        return hashPathHelper(matrix, rows, cols, path[1:], i-1, j, visited)
    # 向左搜索
    elif j - 1 >= 0 and matrix[i*cols+j-1] == path[0] and visited[i][j-1] == 0:
        return hashPathHelper(matrix, rows, cols, path[1:], i, j-1, visited)
    # 向右搜索
    elif j + 1 < cols and matrix[i*cols+j+1] == path[0] and visited[i][j+1] == 0:
        return hashPathHelper(matrix, rows, cols, path[1:], i, j+1, visited)
    else:
        return False
# print(hashPath("ABCESFCSADEE", 3, 4, 'SEE'))

# 和为S的连续序列
def find_continuous_sequence(tsum):
    if not tsum:
        return []
    res = []
    start, end = 0, 1
    while start < (tsum + 1) // 2:
        cur_sum = helper(start, end)
        if cur_sum == tsum:
            tmp = []
            for i in range(start, end+1):
                tmp.append(i)
            res.append(tmp)
            start += 1
        elif cur_sum < tsum:
            end += 1
        elif cur_sum > tsum:
            start += 1
    return res

def helper(start, end):
    res = 0
    for i in range(start, end+1):
        res += i
    return res
print(find_continuous_sequence(15))

# 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的
def twoNumSum(nums, target):
    # 乘积最小: 两端乘积小于中间的乘积
    # 思路: 左右指针,p1指向开始,p2指向结束
    res = []
    if not nums:
        return res
    start, end = 0, len(nums)-1
    while start < end:  # 终止条件: start小于end,即两个指针不相遇
        cur_sum = nums[start] + nums[end]
        if cur_sum == target:
            res.append(nums[start])
            res.append(nums[end])
            break
        elif cur_sum < target:
            start += 1
        elif cur_sum > target:
            end -= 1
        else:
            pass
    return res
print(twoNumSum([1,2,3,4,5,6,7,8,9], 15))

# 出现1次的数字,其余数字出现两次,2个数字出现1次
def find_number_appear_once(nums):
    # step1: 异或,必然得到非0数字 res
    # setp2: 在res中找最低位非0的位 ix
    # step3: 根据ix是否为1,把数组分为2个数组,每个数组中必然有1个数字出现1次,其余出现2次
    # step4: 对2个子数组异或即可
    res = 0
    for num in nums:
        res ^= num

    ix = 0
    while res & 0x01 == 0:
        res = res >> 1
        ix += 1

    nums1, nums2 = [], []
    for num in nums:
        if num & (0x01 << ix):
            nums1.append(num)
        else:
            nums2.append(num)

    res1 = res2 = 0
    for num in nums1:
        res1 ^= num
    for num in nums2:
        res2 ^= num
    return res1, res2
# print(find_number_appear_once([1,2,3,4,2,3,6,6]))


# 在字符串中找到第一次出现一次的字符
def find_once_str(s):
    # 哈希标查找,这里用数组实现
    hash_table = [0] * 256
    for ch in s:
        hash_table[ord(ch)] += 1

    for ch in s:
        if hash_table[ord(ch)] == 1:
            return ch
    return None
# print(find_once_str("abcdeabd"))


def get_first_num(nums, target):
    start, end = 0, len(nums)-1
    while start <= end:
        mid = start + ((end - start) >> 1)
        if nums[mid] == target:
            if mid == start or nums[mid-1] != target:
                return mid
            else:
                end = mid - 1
        elif nums[mid] < target:
            start = mid + 1
        elif nums[mid] > target:
            end = mid - 1
    return -1

def get_last_num(nums, target):
    start, end = 0, len(nums)-1
    while start <= end:
        mid = start + ((end - start) >> 1)
        if nums[mid] == target:
            if mid == end or nums[mid+1] != target:
                return mid
            else:
                start = mid + 1
        elif nums[mid] < target:
            start = mid + 1
        elif nums[mid] > target:
            end = mid - 1
    return -1

# 统计一个数字在排序数字中出现的次数
def getNumOfK(nums, target):
    # 有序数组查找数字 -> 二分法
    # 这里属于二分的变形,有多个terget,需要找第一个和最后一个
    if not nums:
        return 0

    first_ix = get_first_num(nums, target)
    last_ix = get_last_num(nums, target)
    if first_ix == -1 and last_ix == -1:
        return -1
    return last_ix - first_ix + 1
# print(getNumOfK([1,2,3,3,3,3,3,3,4,5], 3))


def isBalanceTreeHelper(root):
    if not root:
        return 0
    left = isBalanceTreeHelper(root.left)
    right = isBalanceTreeHelper(root.right)
    if left == -1 or right == -1:
        return -1
    return -1 if abs(left-right)>1 else 0

# 是否平衡树
def isBalanceTree(root):
    if not root:
        return True
    return isBalanceTreeHelper(root) == -1

# 扑克牌顺子
# 方法1: 计算大小王个数和间隔数,间隔数大于大小王数,则不是顺子
# 方法2: 最大值-最小值必须小于等于4

# 圆圈中剩余的最后数字
def remainNum(n, m):
    if n <= 1 or m <= 0:
        return -1
    people = list(range(1, n+1))
    ix = 0
    while len(people) > 1:
        ix = (ix + m - 1) % len(people)
        del people[ix]
    return people[0]
# print(remainNum(5, 3))

# 左旋转字符串:字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”
def leftRotateStr(s, n):
    # 方法1:
    # n = n % len(s)
    # return s[n:] + s[:n]

    # 方法2: 先反转前n个,后len(s)-n个,然后整体反转
    if not s or n <= 0:
        return s
    s = list(s)

    def reverse(s, start, end):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1

    reverse(s, 0, n-1)
    reverse(s, n, len(s)-1)
    reverse(s, 0, len(s)-1)
    return ''.join(s)
# print(leftRotateStr("abcXYZdef", 3))

# 求数组中的逆序对；逆序对是前面数大于后面数，比如(5,2)是一个逆序对
class Solution:
    def __init__(self):
        self.count = 0

    def reverseOrder(self, nums):
        if len(nums) < 2:
            return 0

        self.merge_sort(nums)
        return self.count

    def merge_sort(self, nums):
        # 归并,两两划分,找子数组间的逆序对
        if len(nums) == 1:
            return nums
        mid = len(nums) >> 1
        left = self.merge_sort(nums[:mid])
        right = self.merge_sort(nums[mid:])

        i = j = 0
        res = []
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
                self.count += len(left[i:])
        res += left[i:]
        res += right[j:]
        return res

# obj = Solution()
# print(obj.reverseOrder([1, 2, 3, 4, 5, 6, 7, 0]))

# 反转单词顺序
# "i am a student" -> "i ma a tneduts"
def reverse(s, start, end):
    while start < end:
        s[start], s[end] = s[end], s[start]
        start += 1
        end -= 1
    return ''.join(s)

def reverseSent(s):
    if not s:
        return ""
    # 翻转整个句子
    s = list(s)
    s = reverse(s, 0, len(s)-1)

    # 翻转单词
    res = []
    words = s.split()
    for word in words:
        res.append(reverse(list(word), 0, len(word)-1))
    return ' '.join(res)
# print(reverseSent("i am a student"))

# topK问题
import heapq
class Solution:
    def __init__(self, k):
        self.data = []
        self.k = k

    def push(self, item):
        if len(self.data) < self.k:
            heapq.heappush(self.data, item)  # 压入item,构成小顶堆
        else:
            if self.data[0] < item:
                heapq.heapreplace(self.data, item)  # 弹出堆顶元素,压入item

    def topk(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]  # 从最大开始显示
obj = Solution(3)
list_num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for num in list_num:
    obj.push(num)
# print(obj.topk())


