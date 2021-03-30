#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/20 17:02
# Author: Hou hailun


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# 标题1: 股票买入卖出
# 先找到迄今为止最小价格，然后在以后找最大利润 = 卖出价格 - 最小价格
def getMaxProfit(prices):
    if not prices:
        return None
    max_profit = float('-inf')
    min_price = float('inf')
    for price in prices:
        if price < min_price:
            min_price = price
        if price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit


# 减小if判断次数
# 找到当前最小价格时，不做利润计算
def getMaxProfit(prices):
    if not prices:
        return None
    max_profit = float('-inf')
    min_price = float('inf')
    for price in prices:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit


# 动态规划
# dp[i]: 第i天卖出的最大收益
# dp[i] = max(dp[i-1], prices[i] - min(prices[:i]))
def getMaxProfit(prices):
    dp = [0]
    for i in range(1, len(prices)):
        dp[i] = max(dp[i - 1], prices[i] - min(prices[0:i]))

    return max(dp)


# 进阶：给出买入卖出的天
def getMaxProfit(prices):
    if not prices:
        return None
    buy_ix = sell_ix = 0
    max_profit = float('-inf')
    min_price = float('inf')
    for ix, price in enumerate(prices):
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
            sell_ix = ix  # 卖出日是最大利润日

    # 买入日是卖出日前最小价格日
    buy_ix = prices.index(min(prices[0:sell_ix]))
    return max_profit


getMaxProfit([2, 4, 1, 2])

# 标题2: 有序数组，2数之和
#  nums = [2, 7, 11, 15], target = 9
def get_two_num_sum(nums, target):
    if not nums:
        return None, None
    nums_len = len(nums)
    if nums_len == 1:
        return None, None

    num1, num2 = 0, nums_len - 1
    while num1 != num2:
        print(num1, num2)
        if nums[num1] + nums[num2] > target:
            num2 -= 1
        elif nums[num1] + nums[num2] < target:
            num1 += 1
        else:
            return num1, num2

    return None, None


# res = get_two_num_sum([2, 7, 11, 15], 9)
# print(res)

# 标题3:非有序数组，如何查找2数之和？
# 哈希表
def get_two_num_sum_hash(nums, target):
    if not nums:
        return None, None
    nums_len = len(nums)
    if nums_len == 1:
        return None, None

    hash_table = {}
    for ix, num in enumerate(nums):
        rest = target - num
        if rest in hash_table.values():
            rest_ix = nums[rest]
            if rest_ix != ix:
                return ix, rest_ix
        hash_table[ix] = num

# res = get_two_num_sum_hash([2,11,7,6], 9)
# print(res)

# 标题4:两数相加
# 两个链表，元素相加，得到一个新的链表
# 输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
# 输出：7 -> 0 -> 8
# 原因：342 + 465 = 807
def two_sum_list(l1, l2):
    if not l1 or not l2:
        return l1 or l2

    new_head = Node(None)  # 头节点
    cur_head = new_head
    carry = 0  # 标记是否进位
    while l1 or l2:  # 循环终止条件: 两个都遍历结束
        num1 = l1.data if l1 else 0
        num2 = l2.data if l2 else 0
        res = num1 + num2 + carry

        node = Node(res % 10)
        cur_head.next = node
        cur_head = node

        carry = res // 10

        # 链表后移
        if l1: l1 = l1.next
        if l2: l2 = l2.next

    # 最高位有进位
    if carry == 1:
        node = Node(1)
        cur_head.next = node

    return new_head.next


# 题目5：无重复最长子串的长度
def get_max_len_substr_no_deplicated(s):
    # 思路：从前往后遍历，若当前不重复，则取该字符串；否则去除前面的重复字符
    res = ''
    max_len = 0
    for ch in s:
        if ch not in res:  # 不重复
            res += ch
            max_len = max(max_len, len(res))
        else:
            res += ch
            ix = res.index(ch)
            res = res[ix+1:]
    return max_len

# 进阶：同时要求获取最长的无重复子串
def get_max_len_substr_no_deplicated_v2(s):

    # step1：最大无重复子串的长度
    max_len = get_max_len_substr_no_deplicated(s)

    # step2：得到最大长度后，设法找到字串
    # 方法1：暴力法，逐个取长度为max_len的非重复字串
    # 方法2：在主串s中找res的位置ix，在ix附近找长度为max_len的非重复子串
    substr = ''
    for i in range(len(s)):
        substr = s[i:i+max_len]
        if len(substr) == len(set(substr)):
            return substr

    return substr

# print(get_max_len_substr_no_deplicated('abcabcbb'))
# print(get_max_len_substr_no_deplicated_v2('abcabcbb'))


# 题目6: 寻找两个有序数组的中位数
def get_median_in_two_sort_array(nums1, nums2):
    # 方法1: 拼接为1个大的有序数组，然后返回中位数

    def solution1(nums1, nums2):
        # 时间复杂度O(m+n), 空间复杂度O(m+n)
        nums = []
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                nums.append(nums1[i])
                i += 1
            else:
                nums.append(nums2[j])
                j += 1
        nums.extend(nums1[i:])
        nums.extend(nums2[j:])

        mid = len(nums) // 2
        if len(nums) % 2 == 1:
            return nums[mid]
        return (nums[mid - 1] + nums[mid]) / 2

    # return solution1(nums1, nums2)

    # 方法2: 有序数组 -> 二分法

nums1 = [1,3,5,7]
nums2 = [2,4,6,8]
# print(get_median_in_two_sort_array(nums1, nums2))


# 标题7:最长回文子串
# 回文串: 正读反读相同
def valid(s, left, right):
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True


def getLongestPalindrome_bp(s):
    # 暴力法，双指针
    s_len = len(s)
    if s_len < 2:
        return False

    res = s[0]
    max_len = 1
    # 枚举所有长度大于等于 2 的子串
    for i in range(s_len - 1):
        for j in range(i + 1, s_len):
            if j - i + 1 > max_len and valid(s, i, j):  # 当前长度大于max_len && 回文串 -> 更新
                max_len = j - i + 1
                res = s[i:j + 1]
    return res

# print(getLongestPalindrome_bp('abcddcba'))


def getLongestPalindrome_dp(s):
    # 动态规划
    # 变量dp[i][j] 表示 s[i,j]是否是回文串
    # 动态转移方程: dp[i][j] = (s[i+1][j-1] and s[i]==s[j]), 表示若s[i+1,j-1]为回文串 且 s[i]==s[j],则s[i,j]也是回文串
    size = len(s)
    if size < 2:
        return False

    dp = [[False for _ in range(size)] for _ in range(size)]

    # 单个字符是回文串,即主对角线上是True
    for i in range(size):
        dp[i][i] = True

    max_len = 1
    start_ix = 0
    # i在左，j在右
    # j 范围 1 ~ size-1
    # i 范围 0 ~ j-1
    for j in range(1, size):
        for i in range(0, j):
            if s[i] == s[j]:
                if j - i < 3:  # i与j不构成边界，至少需要4个
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = False

            if dp[i][j]:
                cur_len = j - i + 1
                if cur_len > max_len:
                    max_len = cur_len
                    start_ix = i

    return s[start_ix: start_ix+max_len]

# print(getLongestPalindrome_dp('abcddcba'))


# 标题8: 生最多水的容器
# 思路：水最多，即面积最大；双指针，left=0，right=size-1,计算面积,然后短的移动，直到left==right为止
def maxArea(heights):
    if not heights:
        return None
    max_area = 0
    left, right = 0, len(heights)-1
    while left < right:
        area = (right - left) * min(heights[left], heights[right])  # 面积 = 长 * 宽
        if area > max_area:
            max_area = area

        # 短的移动
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return max_area

# print(maxArea([1,8,6,2,5,4,8,3,7]))


# 标题9: 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
# 找出所有满足条件且不重复的三元组。
def threeSum(nums):
    # 思路:排序+双指针 (实际上可以认为是三指针)
    size = len(nums)
    if size < 3:
        return None

    res = []
    nums.sort()
    for i in range(size):
        if nums[i] > 0:  # nums[i]>0,则后面的必然也大于0,不存在三数之和等于0的情况
            return res

        # 排序后相邻两数如果相等，则跳出当前循环继续下一次循环，相同的数只需要计算一次
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left = i + 1
        right = size-1
        while left < right:  # 终止条件: left < right
            tmp = nums[i] + nums[left] + nums[right]
            if tmp == 0:
                res.append([nums[i], nums[left], nums[right]])

                # 重复元素跳过
                while left < right and nums[left+1] == nums[left]:
                    left += 1x
                while left < right and nums[right-1] == nums[right]:
                    right -= 1
                left += 1
                right -= 1
            elif tmp < 0:
                left += 1
            else:
                right -= 1

    return res

nums = [-1, 0, 1, 2, -1, -4]
# print(threeSum(nums))

# 标题10: 合并两个有序链表
# 将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的
def mergeList(l1, l2):
    if not l1 or not l2:
        return l1 or l2

    new_head = tmp = Node(None)
    while l1 and l2:  # 终止条件: 两个链表都没有到末尾
        if l1.data < l2.data:
            tmp.next = l1
            l1 = l1.next
        else:
            tmp.next = l2
            l2 = l2.next
        tmp = tmp.next

    if l1:
        tmp.next = l1
    if l2:
        tmp.next = l2

    return new_head.next

# 标题11: 搜索旋转排序数组
# 数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] ,查找给定值
def search(nums, target):
    # 思路: 旋转排序数组,基本有序,采用二分法
    # 本题重点在于区分ix属于前半数组还是后半数组
    #   中间值属于前半数组: mid值大于start值
    #   中间值属于后半数组: mid值小于end值
    if not nums:
        return None
    start, end = 0, len(nums)-1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < nums[end]:  # mid属于后半子数组,在右边查找
            if nums[mid] < target <= nums[end]:  # 右半子数组
                start = mid + 1
            else:
                end = mid - 1
        else:  # mid属于前半子数组,在左边查找
            if nums[start] <= target < nums[end]:  # 左边查找
                end = mid - 1
            else:
                start = mid + 1

    return -1

# print(search([4,5,6,7,0,1,2], 0))

# 标题12: 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置
# 本题属于二分法的变体
def searchRange(nums, target):
    if not nums:
        return None
    low, high = 0, len(nums)-1
    while low < high:
        mid = low + (high - low) // 2
        if nums[mid] == target:
            left = mid
            right = mid
            while left >= 0 and nums[left] == nums[left-1]:
                left -= 1
            while right <= high and nums[right] == nums[right+1]:
                right += 1
            return [left, right]
        elif nums[mid] > target:
            high = mid - 1
        else:
            low = mid + 1
    return None

# print(searchRange([1,2,3,3,3,3,3,4,5], 3))

# 标题13: 组合总数
# 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合
def combineSum(nums, target):
    if not nums:
        return None

# 标题14: 最大子序列和
def maxSubArraySum(nums):
    if not nums:
        return None
    # 思路: 当前和为负数,则跳过当前和,把下一个数字作为新的起始数;当前和为正数,相加;更新最大和
    max_sum = cur_sum = 0
    for num in nums:
        if cur_sum > 0:  # 当前和大于0,则累计
            cur_sum += num
        else:            # 当前和小于等于0,则把下一个数字作为当前和
            cur_sum = num
        if cur_sum > max_sum:  # 更新最大和
            max_sum = cur_sum
    return max_sum

# print(maxSubArraySum([-2,1,-3,4,-1,2,1,-5,4]))

# 标题15: 子集
# 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
def subsets(nums):
    if not nums:
        return None

    res = list()
    res.append([])
    for num in nums:
        new = [s + [num] for s in res]
        res += new
    return res

# print(subsets([1,2,3]))

# 标题16: 对称二叉树
# 先序遍历,从跟节点出发,左子树等于右子树
def isSymmetric_helper(l_root, r_root):
    if l_root is None and r_root is None:  # 左右子树都遍历完成
        return True
    if l_root is None or r_root is None:
        return False
    if l_root.data != r_root.data:
        return False
    return isSymmetric_helper(l_root.left, r_root.right) and isSymmetric_helper(l_root.right, r_root.left)

def isSymmetric(root):
    if not root:
        return True
    return isSymmetric_helper(root.left, root.right)


def isSymmetric_loop(root):
    if not root:
        return True
    stack = [(root.left, root.right)]
    while stack:
        l, r = stack.pop()  # 同时获取左右子节点
        if l is None or r is None:
            if l != r:  # 一个为空,一个不为空
                return False
        else:
            if l.data != r.data:  # 值不相同
                return False
            # 成对append
            stack.append((l.left, r.right))
            stack.append((l.right, r.left))

    return True

# 标题17: 二叉树的最大深度
def maxDepth(root):
    if not root:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    return max(left_depth, right_depth) + 1

def maxDepth_loop(root):
    # 循环版本: 层次遍历
    stack = [root]
    depth = 1
    while stack:
        # 和层次遍历区别,每次都弹出一层的节点
        n = len(stack)
        for _ in range(n):
            node = stack.pop(0)
            stack.append(node.left if node.left else [])
            stack.append(node.right if node.right else [])
        depth += 1
    return depth


# 标题18: 路径总和
# 判断是否存在路径和等于给定值
def hasPathSum(root, sum):
    if not root:
        return False

    # 叶子节点 && 叶子节点值等于剩余值
    if root.left is None and root.right is None:
        return root.val == sum

    # 递归遍历
    return hasPathSum(root.left, sum-root.val) or \
           hasPathSum(root.right, sum-root.val)

# 标题19: 路径总和II
# 找到所有从根节点到叶子节点路径总和等于给定目标和的路径。
def findPathSum(root, sum):
    res = []
    if not root:
        return res

    def helper(root, tmp, sum):
        if root.left is None and root.right is None and root.val == sum:
            tmp.append(root)
            res.append(tmp)

        helper(root.left, tmp+[root.val], sum-root.val)
        helper(root.right, tmp+[root.val], sum-root.val)

    helper(root, [], sum)
    return res

# 标题20: 卖卖股票最佳时机
def maxProfit(nums):
    if not nums:
        return None
    min_price = float('inf')
    max_profit = 0
    for price in nums:
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
    return max_profit

# 进阶:给出买入卖出的天
def maxProfit_v2(nums):
    if not nums:
        return None
    min_price = float('inf')
    max_profit = 0
    buy_ix = sell_ix = 0
    for ix, price in enumerate(nums):
        if price < min_price:
            min_price = price
        elif price - min_price > max_profit:
            max_profit = price - min_price
            sell_ix = ix
    # 买入日是卖出日前最小价格的日期
    buy_ix = nums.index(min(nums[:sell_ix]))
    return max_profit, buy_ix, sell_ix

# print(maxProfit([3, 1, 4, 5]))
# print(maxProfit_v2([3, 1, 4, 5, 3, 0, 2]))

# 标题21: 给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素
def findNumApperenceOnce(nums):
    # 异或: 相同为0,不同为1; 0 ^ x = x
    if not nums:
        return None
    res = 0
    for num in nums:
        res ^= num
    return res


def findNumApperenceOnce_v2(nums):
    # 数学公式
    threshold = 2

    return threshold * sum(set(nums)) - sum(nums)

# print(findNumApperenceOnce([1,2,3,1,3]))
# print(findNumApperenceOnce_v2([1,2,3,1,3]))


# 标题22: 进阶: 给定一个非空整数数组，除了两个元素只出现一次以外，其余每个元素均出现两次
def findTwoNumApperenceOnce(nums):
    # 异或: 相同为0,不同为1; 0 ^ x = x
    if not nums or len(nums) < 2:
        return None, None

    # step1: 全部元素异或
    res = 0
    for num in nums:
        res ^= num

    # step2: 找res中不为0的最低为
    ix = 0
    while not res & (0x01 << ix):
        ix += 1

    # step3: 根据ix位是否等于1,分为两个数组
    nums1, nums2 = [], []
    for num in nums:
        if num & (0x01 << ix):
            nums1.append(num)
        else:
            nums2.append(num)

    # step4: 分别在每个数组中异或计算出现一次的数
    res1 = res2 = 0
    for num in nums1:
        res1 ^= num
    for num in nums2:
        res2 ^= num

    return res1, res2

# print(findTwoNumApperenceOnce([1,2,3,4,2,3]))

# 标题24: 环形链表: 给定一个链表，判断链表中是否有环。
# 前后指针,快指针每次走2步,慢指针每次走1步,若有环,则必然相遇
def hasCircle(head):
    if not head or not head.next:  # 空节点 or 1个节点
        return False
    slow = head
    fast = head.next
    while fast and fast.next:  # 终止条件: fast为空: 已经遍历完所有节点; fast.next为空: 最后一个节点
        slow = slow.next
        fast = fast.next.next
        if fast == slow:
            return True
    return False

# 标题25: 最小栈: o(1)时间内得到当前栈的最小值
class MinStack:
    def __init__(self):
        self.stack = []         # 数据栈
        self.stack_helper = []  # 辅助栈,保存当前数据栈中的最小元素
        self.size = 0

    def push(self, item):
        # 入栈
        self.stack.append(item)

        # 空栈 or 辅助栈中最小元素 大于 item
        if not self.stack_helper or self.stack_helper[-1] > item:
            self.stack_helper.append(item)
        else:
            self.stack_helper.append(self.stack_helper[-1])

        self.size += 1

    def pop(self):
        if self.size == 0:
            return
        self.stack.pop()
        self.stack_helper.pop()
        self.size -= 1

    def top(self):
        if self.size == 0:
            return
        return self.stack[self.size-1]

    def getMin(self):
        if self.size == 0:
            return
        return self.stack_helper[self.size-1]

# 标题26: 多数元素: 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
def majorityElement(nums):
    # 方法1: 排序后,找中位数即可
    # 方法2: 哈希
    # 方法3: 当前元素与下一个元素相同,则累计次数,不同则次数减1,当次数为0时,重置次数和当前元素
    res = nums[0]
    cnt = 1
    for num in nums[1:]:
        if res == num:
            cnt += 1
        else:
            cnt -= 1

        if cnt <= 0:
            res = num
            cnt = 1
    return num

# print(majorityElement([1,2,3,2,2,3,2]))

# 标题27: 相交链表: 本题目要求找两个链表的公共节点
# 方法1: 你走过我的路,我走过你的路,最后我们便相遇了
def findCommonNodeLink(head1, head2):
    if not head1 or not head2:
        return None

    h1, h2 = head1, head2
    while h1 != h2:  # 终止条件: 两指针相遇
        h1 = h1.next if h1 else head2  # h1不为空,继续往后走;走到最后,则沿着head2走
        h2 = h2.next if h2 else head1
    return h1

# 方法2:快慢指针
# 快的指针先走 abs(len(head1)-len(head2)), 然后一起走
def findCommonNodeLink_v2(head1, head2):
    if not head1 or not head2:
        return None

    len_head1 = len_head2 = 0
    node1, node2 = head1, head2
    while node1:
        len_head1 += 1
        node1 = node1.next
    while node2:
        len_head2 += 1
        node2 = node2.next

    fast = head1 if len_head1 > len_head2 else head2
    slow = head2 if len_head1 > len_head2 else head1

    for i in range(abs(len_head1 - len_head2)):
        fast = fast.next

    while fast != slow and fast and slow:  # 终止条件: 相遇 or 有一个遍历完也没有相遇
        fast = fast.next
        slow = slow.next
        if fast == slow:
            return fast

    return None

# 标题28: 打家劫舍
# 相邻房间不能都被偷窃,那么如何获取最大金额
# 动态规划: dp[i]表示从前i间房子所能抢到的最大金额, Ai表示第i间房子的金额
# 只有1个房间,dp[1] = A[1]
# 有2个房间,dp[2] = max(A[1], A[2])
# 有3个房间,dp[3] = max(dp[1]+A[3], dp[2]),即抢第3家,金额累加;步枪第三家
# ...
# 有n个房间,dp[n] = max(dp[n-2]+A[n], dp[n-1])
def rob(nums):
    if not nums:
        return None

    _len = len(nums)
    dp = [None] * _len
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, _len):
        dp[i] = max(nums[i]+dp[i-2], dp[i-1])

    return dp[-1]

# print(rob([1,2,3,3]))

# 标题29:反转链表
def reverseLink(head):
    # 三指针
    if not head or not head.next:
        return None

    pre = Node(None)
    cur = head
    while cur:
        last = head.next
        cur.next = pre
        pre = cur
        cur = last
    return cur.next

# 标题30: 回文链表
# 方法1: 遍历链表,记录链表各个节点元素,判断是否回文
def palindromeList(head):
    if not head or not head.next:
        return True
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res == res[::-1]

# 方法2: 头尾指针, 若不等,则不为回文串; 相等则移动指针直到相遇
# note: 尾指针需要反转链表

# 标题31: 移动零: 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
# 输入: [0,1,0,3,12]
# 输出: [1,3,12,0,0]

# 方法1: 辅助list, 先把nums中非零元素插入到辅助list,然后再写0
def moveZero(nums):
    res = []
    for num in nums:
        if num != 0:
            res.append(num)
    res += [0] * (len(nums) - len(res))
    return res

# 方法2:
# 题目要求：必须在原数组上操作，不能拷贝额外的数组，因此上述方法不适用
# 思路2：双指针
def moveZero_v2(nums):
    j = 0  # j表示非0元素下标
    for i in range(len(nums)):
        if nums[i] != 0:  # i位置元素不等0,则写到j为止
            nums[j] = nums[i]
            j += 1

    # j位置后填充0
    while j < len(nums):
        nums[j] = 0
        j += 1
    return nums

nums = [0,1,0,3,12]
# print(moveZero(nums))
# moveZero_v2(nums)
# print(nums)

# 标题32: 路径总和III
# 给定一个二叉树，它的每个结点都存放着一个整数值。
# 找出路径和等于给定数值的路径总数。
# 路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
# 二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。
# 示例：
# root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
#       10
#      /  \
#     5   -3
#    / \    \
#   3   2   11
#  / \   \
# 3  -2   1
# 返回 3。和等于 8 的路径有:
# 1.  5 -> 3
# 2.  5 -> 2 -> 1
# 3.  -3 -> 11

# 首先解读题干，题干的要求是找和为sum的路径总数，这次路径的起点和终点不要求是根结点和叶结点，
# 可以是任意起终点，而且结点的数值有正有负，但是要求不能回溯，只能是从父结点到子结点的。
# 在已经做了路径总和一和二的基础上，我们用一个全局变量来保存路径总数量，在主调函数中定义变量self.result=0。
# 因为数值有正有负，所以在当我们找到一条路径和已经等于sum的时候，不能停止对这条路径的递归，
# 因为下面的结点可能加加减减，再次出现路径和为sum的情况，因此当遇到和为sum的情况时，
# 只需要用self.result+=1把这条路径记住，然后继续向下进行即可。
# 由于路径的起点不一定是根结点，所以需要对这棵树的所有结点都执行一次搜索，
# 就是树的遍历问题，每到一个结点就执行一次dfs去搜索以该结点为起点的路径：
class Solution:
    def pathSum(self, root, sum):
        self.account = 0

        def throughRoot():
            if root is None:
                return
            sum -= root.val
            if sum == 0:    # 找到路径后,记录个数,继续往下递归查找
                self.account += 1
            threeSum(root.left, sum)
            threeSum(root.right, sum)

        def loop(root, sum):
            if root is None:
                return self.account
            throughRoot(root, sum)
            loop(root.left, sum)  # 每个节点都要作为起始节点
            loop(root.right, sum)

        loop(root, sum)
        return self.account

# 标题33: 找到所有数组中消失的数字
# 给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。
# 找到所有在 [1, n] 范围之间没有出现在数组中的数字。

# 方法1: 利用辅助队列 a = [i for i in range(1, n+1)], 然后查找不在a中的元素
# 方法2: 利用辅助队列 a = [i for i in range(1, n+1)], set(a) - set(nums)
# 方法3: 假设无重复,则i元素必然再位置i上,即i == num[i]
#       如果i != nums[i], 则交换 nums[nums[i]],nums[i]
# 方法4: 假设五重读, 则元素i必然等于nums.index(i)

# 二进制中1的个数
def oneBitNum(n):
    if not n:
        return None
    cnt = 0
    while n:
        n = n & (n-1)
        cnt += 1
    return cnt
# print(oneBitNum(15))

# 标题34: 汉明距离
# 两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目
# 方法:先异或 在判断二进制中1的个数
def hanmingDist(num1, num2):
    num = num1 ^ num2
    return oneBitNum(num)

# print(hanmingDist(12, 15))

# 标题35: 把二叉搜索树转换为累加树
# 给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，使得每个节点的值是原来的节点值加上所有大于它的节点值之和。
# 例如：
# 输入: 二叉搜索树:
#               5
#             /   \
#            2     13
#
# 输出: 转换为累加树:
#              18
#             /   \
#           20     13

# 思路: 二叉搜索树的根节点值大于左孩子节点值,小于右孩子节点值
#   可以从 右-跟-左 顺序遍历, 把当前节点值和上一个节点值相加
num = 0  # 记录上一个节点值
def addTree(root):
    if not root:
        return

    addTree(root.right)

    global num
    root.val = root.val + num
    num = root.val

    addTree(root.left)
    return root

# 标题36: 二叉树的直径
# 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点
# 思路: 直径和左右子树高度有关,最大直径 = 最大左右子树高度相加
class Solution:
    def __init__(self):
        self.max = 0

    def diameterOfBinaryTree(self, root):
        self.depth(root)
        return self.max

    def depth(self, root):
        if not root:
            return 0

        l = self.depth(root.left)
        r = self.depth(root.right)
        self.max = max(self.max, l+r)  # 最大直径 = max(当前最大直径, 左子树高度+右子树高度)
        return 1 + max(l, r)  # 返回当前树高度

# 标题37: 最短无序连续子数组
# 给定一个整数数组，你需要寻找一个连续的子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
# 你找到的子数组应是最短的，请输出它的长度
def findUnsortedSubarray(nums):
    # 思路: 拷贝一份nums后排序;将2排序后的数组和未排序的数组对比,记录下相同值的索引
    if not nums or len(nums) == 1:
        return nums
    nums_copy = nums[:]
    nums_copy.sort()

    start_flag = end_flag = True
    start, end = -1, len(nums)-1
    for i in range(len(nums)):  # 遍历一次,同时记录相同值的索引
        if nums_copy[i] == nums[i] and start_flag:  # start_flag用来记录第一个遇到不同值时的索引
            start = i
        else:
            start_flag = False
        if nums_copy[-i-1] == nums[-i-1] and end_flag:
            end = len(nums)-i-1
        else:
            end_flag = False

        if end <= start:
            return 0
    return end - start - 1

# print(findUnsortedSubarray([2, 6, 4, 8, 10, 9, 15]))


# 标题38: 合并二叉树
# 给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
# 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。

def mergeTree(root1, root2):
    # 思路: 两棵树同时进行前序遍历
    # case1: 都存在节点,则累加
    # case2: 有1个不存在,则返回另一颗树
    # case3: 都不存在,则任意返回一棵树
    if not root1 or not root2:  # case2, case3
        return root1 or root2

    root1.val += root2.val
    mergeTree(root1.left, root2.left)  # 递归遍历
    mergeTree(root1.right, root2.right)
    return root1

# 合并两个有序数组
# 题目描述：给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
# 说明:
# 初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
# 你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
def mergeSortArray(nums1, nums2, m, n):
    # 方法1: 把nums2写到nums1后面, 在排序
    # 方法2: 从后往前遍历, p1指向num1,p2指向num2. 若p1小于p2,则在len(nums1)+nums[nums2]-1位置写p2;否则写p1;

    while m > 0 and n > 0:  # 终止条件:有1个遍历完成
        if nums1[m-1] < nums2[n-1]:
            nums1[m+n-1] = nums2[n-1]
            n -= 1
        else:
            nums1[m+n-1] = nums1[m-1]
            m -= 1
    # 处理nums2还有元素的可能, 把nums2剩余元素写入nums1
    if n > 0:
        nums1[0:n] = nums2[0:n]
    return nums1

a = [2,5,7,0,0,0]
b = [1,3,8]
# print(mergeSortArray(a, b, 3, 3))


# 找不同
def findTheDifference(s, t):
    # 方法1: set(t)-set(s)
    # 方法2: 异或,由于s,t只有1个字符不同,其余都 相同,因此,两个异或后结果即为所求字符
    res = ord(t[-1])
    for i in range(len(s)):
        res ^= ord(s[i])
        res ^= ord(t[i])
    return chr(res)
# print(findTheDifference('ab', 'bax'))

# 两个数组的交集I:
# 要求: 输出结果中的每个元素一定是唯一的。
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))

# 进阶: 要求输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致
def intersection_ii(nums1, nums2):
    # 方法1: 列表生成式
    res = [i for i in nums1 if i in nums2]
    # 方法2: 哈希标
    return res

# print(intersection([1,2,3, 2,4], [2,2,4]))
# print(intersection_ii([1,2,3, 2,4], [2,2,4]))

# 两整数之和
# 不使用加减乘除,计算两个整数和
# 方法:
#   不考虑进位:两数相加,类似异或,相同为0,不同为1
#   考虑进位: 0 & x = 0, 1 & 1 = 0,产生进位
#   循环执行以上2步,直到没有进位为止
def twoNumSum(num1, num2):
    if num1 == 0 or num2 == 0:
        return num1 or num2

    while num2:
        _sum = num1 ^ num2
        carry = (num1 & num2) << 1  # 进位左移

        num1 = _sum
        num2 = carry
    return num1

# print(twoNumSum(7, 2))

# 给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。
# 案例:
# s = "leetcode"        返回 0.
# s = "loveleetcode",   返回 2.
def findFirstApperenceOnceCh(s):
    # 思路:涉及到次数问题,优先考虑哈希表
    hash_table = {}
    for ch in s:
        if ch in hash_table:
            hash_table[ch] += 1
        else:
            hash_table[ch] = 1

    for ix, ch in enumerate(s):
        if hash_table[ch] == 1:
            return ix, ch
    return None

    # 方法2: count(ch) == 1
# print(findFirstApperenceOnceCh('abcdab'))

# 计算给定二叉树的所有左叶子之和
# 问题: 如何判断是左叶子?
# 回答: if node.left and node.left.left is None and node.left.right is None, 则说明node.left是node的左叶子节点
def leftNodeSum(root):
    if not root:
        return 0
    sum = 0
    if root.left and root.left.left is Node and root.left.right is None:
        sum += root.left.val

    sum += leftNodeSum(root.left)
    sum += leftNodeSum(root.right)
    return sum

# 买卖股票的最佳实际II
# 要求多次买入卖出
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    profit = 0
    i = 0
    while i < len(prices) - 1:  # 循环:后一天价格大于前一天价格,则卖出,累计利润
        if prices[i+1] > prices[i]:
            profit += prices[i+1] - prices[i]
        i = i+1
    return profit
# print(maxProfit([1,2,3]))

# 删除排序数据中的重复项
# 题目描述：给定一个排序数组，你需要在原地删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
# 不要使用额外的数组空间，你必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。
# 示例 1: 给定数组 nums = [1,1,2],
# 函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。
def dropDuplicateArray(nums):
    # 方法1: 遇到重复的删除
    def dropDuplicateArray_v1(nums):
        if not nums:
            return 0
        for i in range(len(nums)):
            while i < len(nums) - 1 and nums[i] == nums[i + 1]:  # 删除重复元素
                nums.pop(i + 1)
        return len(nums)

    # return dropDuplicateArray_v1(nums)

    def dropDuplicateArray_v2(nums):
        # 方法2:不删除,把不重复的写到前面
        slow = 0
        length = len(nums) - 1
        for fast in range(length):
            if nums[fast] != nums[fast+1]:  # 不重复,把fast+1指向的指向的元素写到slow+1指向的为止
                nums[slow+1] = nums[fast+1]  # 注意:写入的fast+1,因为要写入不重复元素
                slow += 1
        return slow + 1

    return dropDuplicateArray_v2(nums)

# print(dropDuplicateArray([1,2,2,3, 4]))

# 加一
# 定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
# 最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。
# 你可以假设除了整数 0 之外，这个整数不会以零开头。
def addOne(nums):
    # 方法1: 把nums数组转换为整数,加1,在转换为列表
    def addOne_v1(nums):
        tmp = 0
        for num in nums:
            tmp = tmp * 10 + num

        tmp += 1
        res = []
        while tmp:
            res.append(tmp % 10)
            tmp = tmp // 10
        return res[::-1]
    # return addOne_v1(nums)

    # 方法2: 从后往前遍历数组,+1后有进位,则加1
    def addOne_v2(nums):
        res = []
        plus = 1
        for num in nums[::-1]:
            v = num + plus
            if v > 9:  # 有进位
                plus = 1
                v = v % 10
            else:
                plus = 0
            res.insert(0, v)
        if plus == 1:  # 最高位进位
            res.insert(0, plus)
        return res
    return addOne_v2(nums)
# print(addOne([1,2,3]))

# 反转整数
# 给定一个 32 位有符号整数，将整数中的数字进行反转。
def reverse(x):
    reverse_num = int(str(abs(x))[::-1])
    if reverse_num.bit_length() > 31:  # 反转后溢出
        return 0
    return reverse_num if x > 0 else -reverse_num

# 回文数
# return x > 0 and str(x) == str(x)[::-1]

# 搜索插入位置
# 题目描述：给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。你可以假设数组中无重复元素。
def searchInsert(nums, target):
    # 方法1: O(NlogN)
    def searchInsert_v1(nums, target):
        if target in nums:
            return nums.index(target)
        nums.append(target)
        return nums.index(target)
    # return searchInsert_v1(nums, target)

    # 方法2: 有序数组, 二分法, 找不到则遍历数组
    def searchInsert_v2(nums, target):
        start, end = 0, len(nums)-1
        mid = start + (end - start) // 2
        while start < end:  # 一般二分法是start<=end, 为毛这里是<呢?
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid - 1
            else:
                start = mid - 1
            mid = start + (end - start) // 2

        print(start, end)
        # 没有找到
        if nums[mid] >= target:
            return mid
        return mid+1

    return searchInsert_v2(nums, target)

# print(searchInsert([1,3,4,5], 2))

# 移除元素: 在数组中删除指定元素
def removeNum(nums, val):
    # 思路: 前后指针, 若fast != val, 则fast写到slow位置
    slow = 0
    length = len(nums) - 1
    for fast in range(length):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1

    return slow
# print(removeNum([1,2,3,4,2,2], 2))

# 题目名称：最长公共前缀
# 题目描述：编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串 ""。
def maxLenCommPreStr(strs):
    # 思路: 最短长度的字符串中找最长公共前缀
    min_str = min(strs, key=len)  # 找出最短字符串
    for ix, ch in enumerate(min_str):
        for other in strs:
            if other[ix] != ch:   # 字符ch不是other的前缀,则返回当前最长前缀
                return min_str[:ix]
    return min_str
# print(maxLenCommPreStr(['abc', 'abcde', 'ae']))

# 标题:实现strStr
def strStr(haystack, needle):
    # return haystack.fing(needle)

    # 朴素法
    def strStr_bp(haystack, needle):
        i = j = 0
        while i < len(haystack) and j < len(needle):
            if haystack[i] == needle[j]:  # 匹配成功
                i += 1
                j += 1
            else:   # 匹配失败
                i = i - j + 1   # i回退到上次匹配开始位置的下一位置
                j = 0
        if j == len(needle):  # 匹配成功
            return i - j      # 返回起始位置
        return -1
    # return strStr_bp(haystack, needle)

    # 每次在haystack中找长度为len(needle)的字符,比较是否相同
    def strStr_cut(haystack, needle):
        _len = len(needle)
        for i in range(len(haystack) - _len + 1):
            if haystack[i: i + _len] == needle:
                return i
        return -1
    return strStr_cut(haystack, needle)

# print(strStr('abcde', 'cde'))

# 题目名称：最后一个单词的长度
# 思路: 首先把句子拆分为单词列表  然后,求最后1个单词的长度
def lengthOfLastWord(s):
    if not s:
        return 0
    words = s.split()
    if words:
        return len(words[-1])
    return 0
# print(lengthOfLastWord('hello word! python '))

# 二进制加1
# 思路: 先转换为10进制,求和后,在转为2进制
# 类似题目: 异或 求累加, 位于求进位
def binaryAdd(a, b):
    a_int = int(a, 2)
    b_int = int(b, 2)

    res = a_int + b_int

    return bin(res)[2:]
# print(binaryAdd('11', '1'))

# 反转字符串
def reverseStr(s):
    # 方法1: return s[::-1]
    # 方法2: 双指针法
    start, end = 0, len(s)-1
    while start < end:
        s[start], s[end] = s[end], s[start]
        start += 1
        end -= 1
    return s
# print(reverseStr(['a', 'b', 'c', 'd']))

 # 字符串相加
# 给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和
def strAdd(s1, s2):
    # 思路: 先转换为十进制,然后相加后再转换为字符串
    # return str(int(s1) + int(s2))
    # 问题:字符串很大时,对于C++等语言int后会造成溢出(python不存在溢出问题)

    # 方法2:
    s1_list = list(s1)
    s2_list = list(s2)
    sum1 = sum2 = 0
    for ch in s1_list:
        sum1 = sum1 * 10 + int(ch)
    for ch in s2_list:
        sum2 = sum2 * 10 + int(ch)
    return str(sum1 + sum2)
# print(strAdd('12', '12'))

# 压缩字符串
# input: ["a","a","b","b","c","c","c"]
# 输出：返回6，输入数组的前6个字符应该是：["a","2","b","2","c","3"]
def compress(chars):
    if not chars:
        return None
    # 方法1: set(chars)得到有序无重复元素,然后逐个找元素的次数
    def compress_v1(chars):
        ch_set = set(chars)
        res = []
        for ch in ch_set:
            count = chars.count(ch)
            if count == 1:
                res.append(ch)
            else:
                res.append(ch)
                res = res + list(str(count))
        return res
    # return compress_v1(chars)

    # 方法2: 不使用辅助set,count内置函数,从后往前,统计字符个数
    def compress_v2(chars):
        count = 1
        res = []
        for i in range(len(chars)-1, -1, -1):
            if chars[i] == chars[i-1]:
                count += 1
            else:
                res += [chars[i]] if count == 1 else [chars[i]]+list(str(count))
                count = 1
        return res[::-1]
    return compress_v2(chars)
# print(compress(["a","a","b","b","c","c","c"]))

# 541: 翻转字符串II
def reverseStr_ii(s, k):
    # 因为list切片超出索引后为空列表,不影响后续结果
    start, mid, end = 0, k, 2 * k
    res = ''
    while len(res) < len(s):
        res += s[start:mid][::-1] + s[mid:end]
        start, mid, end = start + 2 * k, mid + 2 * k, end + 2 * k
    return res
# print(reverseStr_ii('abcdefgfxy', 2))

# 重复的子字符串
# 给定一个非空的字符串，判断它是否可以由它的一个子串重复多次构成
def duplicatedStr(s):
    # 方法1: 问题:set(s)不是按照先后顺序
    def duplicatedStr_v1(s):
        sub_s = ''.join(list(set(s)))
        s_len, sub_s_len = len(s), len(sub_s)
        tmp = sub_s * (s_len // sub_s_len)
        if tmp == s:
            return True
        return False
    # return duplicatedStr_v1(s)

    def duplicatedStr_v2(s):
        # 逐个比较长度从1到len(s)-1的字符串
        s_len = len(s)
        for i in range(1, s_len):
            times = s_len // i
            if s[:i] * times == s:
                return True
        return False
    # return duplicatedStr_v2(s)

    def duplicatedStr_v3(s):
        return s in (s+s)[1:-1]
    return duplicatedStr_v3(s)

# print(duplicatedStr('ababc'))

# 翻转字符串中的单词III
def reverseWords(s):
    if not s:
        return s
    # words = s.split()
    # res = ''
    # for word in words:
    #     res += word[::-1]
    #     res += ' '
    # return res[:-1]

    # 一句话优化
    return ' '.join([word[::-1] for word in s.split()])

# print(reverseWords("Let's take LeetCode contest"))

# 重复叠加字符串匹配
# A重复多少次后,B是A的字串
def repeatStrMatch(A, B):
    r = A
    cnt = 1
    # 不断累加A,直到r的长度大于等于B的长度,记录累加次数
    while len(r) < len(B):
        r += A
        cnt += 1
    if B in r:  # 若B是r的字串,直接返回cnt
        return cnt
    r += A  # 由于头尾没接上,导致B不是r的字串,再加A,解决头尾未接上问题
    cnt += 1
    if B in r:
        return cnt
    return -1
A = "abcd"
B = "cdabcdab"
# print(repeatStrMatch(A, B))

# 最常见的单词
import re
from collections import Counter
def mostCommonWord(paragraph, banned):
    paragraph = paragraph.lower()
    words = re.findall(pattern='[a-zA-Z]+', string=paragraph)
    word_count = Counter(words)
    res = word_count.most_common(len(banned) + 1)
    for word_cnt in res:
        if word_cnt[0] not in banned:
            return word_cnt
# print(mostCommonWord("b12,b,b,c", ['a']))

# 亲密字符串
# 给定两个由小写字母构成的字符串 A 和 B ，只要我们可以通过交换 A 中的两个字母得到与 B 相等的结果，就返回 true ；否则返回 false 。
def bubbfyString(A, B):
    # 必须长度相同
    if len(A) != len(B):
        return False
    # 两字符串相等, 且A有重复元素  A = aab, B = aab
    if A == B and len(set(A)) < len(B):
        return True
    # A,B字符串中按顺序不相等对数只能为2对，且要求对称
    # [(a, b), (b, a)]
    diff = [(a, b) for a, b in zip(A, B) if a != b]
    return len(diff) == 2 and diff[0] == diff[1][::-1]

# 数组中重复的数字
# 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
def findRepeatNumber(nums):
    # 方法1: python count()
    # 方法2: 哈希表
    # 方法3: 利用不重复,则i == nums.index(i)
    if not nums or len(nums) == 1:
        return None

    for i in range(len(nums)):
        while i != nums[i]:  # i位置 不等于 i位置的元素, 需要把nums[i]元素放到nums[i]位置上
            if nums[i] == nums[nums[i]]:  # 如果nums[i]位置上有元素,则说明nums[i]是重复元素
                return nums[i]

            temp = nums[i]
            nums[i], nums[temp] = nums[temp], nums[i]
    return None
# print(findRepeatNumber([2,3,1,0,2,5,3]))

# 二维数组的查找
# 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
def findNum(nums, target):
    if not nums:
        return False
    m, n = len(nums), len(nums[0])
    i, j = 0, n-1
    while i < m and j >= 0:
        if nums[i][j] == target:
            return True, i, j
        elif nums[i][j] > target:
            j -= 1
        elif nums[i][j] < target:
            i += 1
        else:
            pass
    return False
# print(findNum([[1, 2, 3, 5], [2, 3, 5, 7], [4, 5, 7, 9]], 7))

# 替换空格
def replaceBlank(s):
    # 方法1: return s.replace(' ', '%20')
    # 方法2: 遍历,遇到空格替换为%20
    s_list = list(s)
    for i in range(len(s_list)):
        if s_list[i] == ' ':
            s_list[i] = '%20'
    return ''.join(s_list)
# print(replaceBlank('hello python '))

# 从尾到头打印链表
# 思路: 一般都是从前到后打印,本题要求从后往前打印,利用栈的先进后出性质,可以实现
def printListFromTailToHead(head):
    if not head:
        return None
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res[::-1]

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 重建二叉树
# 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# # 例如，给出
# 前序遍历 preorder = [3,9,20,15,7]
# 中序遍历 inorder = [9,3,15,20,7]
def rebuildTree(preorder, inorder):
    if not preorder or not inorder:
        return
    # 思路: 先找根节点,左子树和右子树;然后递归左右子树
    root = TreeNode(preorder[0])  # 根节点为当前前序遍历的第一个值
    loc = inorder.index(preorder[0])  # 在中序遍历中找根节点index

    root.left = rebuildTree(preorder[1:loc+1], inorder[:loc])
    root.right = rebuildTree(preorder[loc+1:], inorder[loc+1:])
    return root

# 两个栈实现队列
class StackMakeQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, item):
        # 入栈: 入栈1
        self.stack1.append(item)

    def push(self):
        # 出栈: 若栈2不为空,则出栈2,否则吧栈1元素同步到栈2后出栈2
        if self.stack2:
            return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop() if self.stack2 else -1

# 斐波那契数列
# a[i] = a[i-1] + a[i-2]
# 方法1: 动态规划
# 方法2: a,b = b, a+b
def fibnacca(n):
    # 求第n个斐波那契数

    def fibnacca_dp(n):
        dp = [0] * (n+1)
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
    # return fibnacca_dp(n)

    def fibnacca_v2(n):
        a, b = 0, 1
        for i in range(n):
            a, b = b, a+b
        return a
    return fibnacca_v2(n)
# print(fibnacca(8))

# 旋转数组的最小数字
# 2个自增子数组, 找最小值,二分法
def minVal(nums):
    if not nums:
        return
    s, e = 0, len(nums)-1
    while e - s > 1:  # 终止条件: 两个下标相差为1
        mid = (s + e) // 2
        if nums[s] == nums[mid] == nums[e]:  # 如果出现大量重复(例如：[1,0,1,1,1])，只能顺序查找
            # 顺序查找
            min_val = float('inf')
            for num in nums[s: e+1]:
                if min_val > num:
                    min_val = num
            return min_val
        elif nums[mid] > nums[s]:  # mid是第一个子数组,则最小值必然在后面
            s = mid
        else:
            e = mid
    return nums[e]
# print(minVal([3,4,5,1,2]))

# 矩阵中的路径
# 思路: 首先在矩阵中查找第一个字符,然后沿着上下左右四个方向,逐个找其他字符,失败的后退,沿着其他方向找

def exist_helper(board, rows, columns, i, j, word, visited):
    if not word:  # 查找完成
        return True

    # 异常情况
    if i < 0 or i >= rows or j < 0 or j >= columns or visited[i][j] == 1 or board[i][j] != word[0]:
        return False

    visited[i][j] = 1
    # 递归 上下左右 方向，如果有一个方向可存在word路径，则返回
    # 不存在，则标记visited[i][j]未访问，便于下次访问
    if (exist_helper(board, rows, columns, i + 1, j, word[1:], visited) or
            exist_helper(board, rows, columns, i - 1, j, word[1:], visited) or
            exist_helper(board, rows, columns, i, j - 1, word[1:], visited) or
            exist_helper(board, rows, columns, i, j + 1, word[1:], visited)):
        return True

    # 回溯，将当前位置的布尔值标记为未访问
    visited[i][j] = 0
    return False


def exist(board, word):
    if not board or not word:
        return False
    rows, cols = len(board), len(board[0])
    visited = [[0 for _ in range(cols)] for _ in range(rows)]

    # 查找首字符位置
    for i in range(rows):
        for j in range(cols):
            if board[i][j] == word[0]:
                return exist_helper(board, rows, cols, i, j, word[1:], visited)
    return False

# 机器人的运功范围

# 剪绳子
# 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m] 。
# 请问 k[0]*k[1]*...*k[m] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18
# 分析:
#   f(n): 表示长度为n的绳子 剪开后的最大长度

# 二进制中1的个数 n = n & (n-1)

# 删除链表的节点
# 思路: 查找待删除节点的前一个节点,然后删除即可,注意删除头节点的情况
def deleteNode(head, val):
    if not head:
        return head

    if head.val == val:  # 删除头节点
        return head.next

    node = head.next
    pre_node = head
    while node:
        if node.val != val:
            pre_node, node = node, node.next
        else:
            pre_node.next = node.next  # 删除node节点
            break
    return head

# 调整数组位置,使得奇数在偶数前
# 方法1: 利用辅助列表,分别保存奇数和偶数,然后拼接
# 方法2: 前后指针p1, p2, p1当遇到奇数时p1++, 直到遇到偶数时;同时p2直到遇到奇数,交换
#       缺点:会改变奇数,偶数的相对顺序
def exchange(nums):
    if not nums:
        return nums

    def exchange_v1(nums):
        l1, l2 = [], []
        for num in nums:
            if num % 2 == 1:
                l1.append(num)
            else:
                l2.append(num)
        return l1+l2
    # return exchange_v1(nums)

    def exchange_v2(nums):
        s, e = 0, len(nums)-1
        while e - s > 1:
            while s < e and nums[s] % 2 == 1:
                s += 1
            while s < e and nums[e] % 2 == 0:
                e -= 1

            nums[s], nums[e] = nums[e], nums[s]
        return nums
    return exchange_v2(nums)
nums = [1,2,3,4,5]
# print(exchange(nums))

# 链表中倒数第K个节点
# 方法: 首先,计算链表的总长度head_len  然后,从头往后走head_len - k步,该节点即为所求节点
def kNode(head, k):
    if not head:
        return None
    node_cnt = 0
    node = head
    while node:
        node_cnt += 1
        node = node.next
    if node_cnt < k:  # 链表长度小于k,则输入错误
        return None

    node = head
    for i in range(node_cnt - k):
        node = node.next
    return node

# 方法: 双指针 - 前后指针
# p1先走k步,然后p1,p2一起走,p1走到最后为None时,p2即为所求
def kNode_v2(head, k):
    p1 = p2 = head
    for i in range(k):
        if p1:
            p1 = p1.next
    while p1:
        p1 = p1.next
        p2 = p2.next
    return p2

# 树的子结构
# 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)
def isSubTree_helper(treeA, treeB):
    if not treeB:  # 树B遍历完成
        return True
    if not treeA:  # 树A完成,树B未完成,则B不是A的子结构
        return False
    if treeA.val != treeB.val:  # 值不同,则B不是A的子结构
        return False
    # 同时递归遍历左右子树, 并且左右子树都是子结构
    return isSubTree_helper(treeA.left, treeB.left) and \
           isSubTree_helper(treeA.right, treeB.right)

def isSubTree(treeA, treeB):
    if not treeA or not treeB:
        return False
    result = False
    # 首先在树A中查找树B的根节点,先序遍历查找
    if treeA.val == treeB.val:  # A,B树根节点相同
        result = isSubTree_helper(treeA, treeB)
    if not result:  # A树左子树查找
        result = isSubTree_helper(treeA.left, treeB)
    if not result:  # A树右子树查找
        result = isSubTree_helper(treeA.right, treeB)
    return result

# 二叉树的镜像
# 请完成一个函数，输入一个二叉树，该函数输出它的镜像
def mirrorTree(root):
    if not root:
        return None
    # 交换左右节点
    root.left, root.right = root.right, root.left
    mirrorTree(root.left)
    mirrorTree(root.right)
    return root

# 最小的K个数
# 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
def GetLeastNumbers(nums, k):
    # 方法1: 排序后取前k个, O(NlogN)
    # 方法2: partition()
    def partition(nums, start, end):
        print('partition(): start:%s, end:%s' % (start, end))
        key = nums[start]
        while start < end:
            while start < end and nums[end] >= key:
                end -= 1
            nums[start], nums[end] = nums[end], nums[start]
            while start < end and nums[start] <= key:
                start += 1
            nums[start], nums[end] = nums[end], nums[start]
        return start

    index = partition(nums, 0, len(nums) - 1)
    while index != k - 1:
        if index < k - 1:  # 在后面找
            index = partition(nums, index + 1, len(nums) - 1)
        if index > k - 1:  # 在前面找
            index = partition(nums, 0, index - 1)
        print(index, nums)
    return nums[:k]

# print(GetLeastNumbers([4,5,1,6,2,7,3,8], 6))

# 数据流中的中位数
# 思路:数据流不断输入,需要持续记录
class GetMedian:
    def __init__(self):
        self.stack = []
        self.size = 0

    def addNum(self, val):
        self.stack.append(val)
        self.size += 1

    def get_median(self):
        if self.size == 0:
            return None
        if self.size % 2 == 1:  # 奇数
            return self.stack[self.size // 2]
        return (self.stack[(self.size -1 ) // 2] + self.stack[(self.size + 1) //2]) / 2

# 连续子数组的最大和
# 和上面题的重复

# 把数字排成最小的数
from functools import cmp_to_key
def minNumber(nums):
    if not nums:
        return None

    def sort_rule(x, y):
        if x+y < y+x:
            return -1
        elif x+y > y+x:
            return 1
        else:
            return 0

    strs = [str(num) for num in nums]
    nums.sort(key=cmp_to_key(sort_rule))
    return ''.join(strs)

# print(minNumber([10, 2]))

# 把数字翻译成字符串

# 给定一个字符串数组，所有字符都连续成对出现除了一个字母，找出该字母。
# 例子：[AABBCCDEEFFGG], 输出为D

def find_once_str(s):
    # 方法1：从前往后，逐个比较i和i+1位置的元素是否相等
    def find_once_str_v1(s):
        for i in range(0, len(s) - 2, 2):
            if s[i] != s[i+1]:
                break
        return s[i]
    # return find_once_str_v1(s)

    def find_one_str_v2(s):
        # 优化算法：二分法
        # 思路：
        if s[0] != s[1]:
            return s[0]
        n = len(s)
        if s[n-1] != s[n-2]:
            return s[n-1]
        start, end = 0, n-1
        while start <= end:
            mid = (start + end) // 2
            if s[mid] == s[mid-1]:
                end = mid - 1
            else:
                if (a[mid] != a[mid - 1] and a[mid] != a[mid + 1]):
                    return a[mid]
                else:
                    start = mid + 2
        return s[start]

print(find_once_str('AABBCCDEEFFGG'))

