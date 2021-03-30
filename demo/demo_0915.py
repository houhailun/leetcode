#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/9/15 10:13
# Author: Hou hailun


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def findDuplicatesNums(nums):
    # 长度为n的数组中，数字范围为0~n-1, 现在不确定是否有重复数字，请找出重复数字
    if not nums:
        return None

    def solution_hash(nums):
        hash_table = dict.fromkeys(nums, 0)
        for num in nums:
            if hash_table[num] == 0:
                hash_table[num] += 1
            else:
                return num
    # return solution_hash(nums)

    def solution_better(nums):
        # 方法：如果没有重复，则数字等于其在数组中的下标；因此可以如果数值不等于下标，则置换
        for i in range(len(nums)):
            while i != nums[i]:
                if nums[i] == nums[nums[i]]:  # 重复
                    return nums[i]
                tmp = nums[i]
                nums[tmp], nums[i] = nums[i], nums[tmp]
    return solution_better(nums)

# print(findDuplicatesNums([0, 1, 2, 3, 2]))


# 二维数组中的查找：从左上角开始找，若小于key，则往下；若大于key，则往左
# 替换空格
#   方法1：replace(' ', '%20')
#   方法2：把str转为list，然后对list中的空格进行替换，最后转换为str
# 从尾到头打印链表
#   方法1：转换链表指针
#   方法2：栈
#   方法3：递归法

# 重建二叉树：输入前序和中序
# 1、首先构建根节点：前序第一个数
# 2、构建左右子树：在中序中区分左子树和右子树，然后递归创建
def rebuild_tree(pre_input, in_input):
    if not pre_input or not in_input:
        return None

    root = TreeNode(pre_input[0])
    index = in_input.index(pre_input[0])
    root.left = rebuild_tree(pre_input[1: index+1], in_input[:index])
    root.right = rebuild_tree(pre_input[index+1:], in_input[index+1:])
    return root


# 两个栈实现队列
class TwoStackZMakeQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, item):
        # 入队列：栈1
        self.stack1.append(item)

    def pop(self):
        # 出队列：出栈2，那么如何出栈2后实现队列先进先出呢？
        # 把栈1弹出压入栈2，这样先入栈1的元素被压入栈2的栈底，弹出栈2时，栈底的最后弹出
        if self.stack2:
            return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        if self.stack2:
            return self.stack1.pop()
        raise Exception('stack is empty')

# 两个队列实现栈
class TwoQueueMakeStack:
    def __init__(self):
        self.queue1 = []
        self.queue2 = []

    def push(self, item):
        # 入栈：压入队列1
        self.queue1.append(item)

    def pop(self):
        # 出栈：出队列2，那么如何出队列2后实现栈的先入后出呢？
        # 把队列1的元素除了最后一个外全部压入队列2中，这样队列1中的元素是最后一个压入的，交换队列后出队列2，实现了先入后出
        # 特殊情况：队列1中只有一个元素时，直接交换
        if not self.queue1:
            raise Exception('queue is empty')
        while len(self.queue1) > 1:
            self.queue2.append(self.queue1.pop(0))
        self.queue1, self.queue2 = self.queue2, self.queue1
        return self.queue2.pop()

# 旋转数组的最小数字
# 基本有序 -> 二分法
# start指向前半有序数组最后一个数，end指向后半有序数组的第一个数字
def findMinNumInReverseNums(nums):
    if not nums:
        raise Exception('nums is empty')
    start, end = 0, len(nums)-1
    if nums[end] > nums[start]:
        return None
    while start != end - 1:  # 循环终止条件: start + 1 == end
        mid = start + (end - start) // 2
        if nums[mid] == nums[start] == nums[end]:  # 只能顺序查找
            min_val = nums[start]
            for val in nums[start+1: end+1]:
                if min_val > val:
                    min_val = val
            return min_val
        elif nums[mid] >= nums[start]:  # mid指向的是前半个子数组
            start = mid
        elif nums[mid] <= nums[end]:    # mid指向的是后半个子数组
            end = mid
    return nums[end]
nums = [3,4,5,1,2,3]
# print(findMinNumInReverseNums(nums))

# 矩阵中的路径
# 回溯法
def findPathInMat(board, word):
    if not board or not word:
        return None
    rows = len(board)
    columns = len(board[0])
    visited = [[0] * columns for i in range(rows)]
    for i in range(rows):
        for j in range(columns):
            if findPathInMatHelper(board, word, visited, columns, rows, i, j):
                return True
    return False

def findPathInMatHelper(board, word, visited, columns, rows, i, j):
    if not word:
        return True
    # 异常情况:
    if i < 0 or i >= rows or j < 0 or j >= columns or board[i][j] != word[0] or visited[i][j]==1:
        return False

    visited[i][j] = 1
    if (findPathInMatHelper(board, word[1:], visited, columns, rows, i-1, j) or
        findPathInMatHelper(board, word[1:], visited, columns, rows, i+1, j) or
        findPathInMatHelper(board, word[1:], visited, columns, rows, i, j-1) or
        findPathInMatHelper(board, word[1:], visited, columns, rows, i, j+1)):
        return True

    visited[i][j] = 0
    return False

# 二进制中1的个数


# 在数组中查找2个不重复的数字和等于给定值的下标
def getTwoSum(nums, target):
    if not nums:
        return []
    hash_table = {}
    for ix, num in enumerate(nums):
        res = target - num
        if res in hash_table.values():
            if nums.index(res) != ix:
                return [ix, nums.index(res)]
        else:
            hash_table[ix] = num
    return
# print(getTwoSum([2, 7, 11, 15], 9))
    # 方法2: 如果数据有序，则可以使用前后指针

# 给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
# 如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
def two_sum_link(link1, link2):
    if not link1 or not link2:
        return link1 or link2

    # 头节点
    head = Node(-1)
    node = head
    p1, p2 = link1, link2
    carry = 0
    while p1 or p2:
        x = p1.data if p1 else 0
        y = p2.data if p2 else 0
        cur_sum = x + y + carry
        new_node = Node(cur_sum % 10)
        node.next = new_node
        node = node.next
        carry = cur_sum // 10

        if p1:
            p1 = p1.next
        if p2:
            p2 = p2.next

    if carry == 1:  # 最高位进位
        new_node = Node(1)
        node.next = new_node
    return head.next

# 两数相加不用加法
# 不考虑进位：0+0=0，0+1=1,1+1=0,即相同为0不同为1，等同于异或
# 考虑进位: 0+0=0，0+1=0，1+1进位1，等同于位于操作
def two_sum_no_add(num1, num2):
    while num2:
        _sum = num1 ^ num2
        carry = (num1 & num2) << 1
        num1 = _sum
        num2 = carry
    return num1
# print(two_sum_no_add(5, 6))


# 无重复字符的最长子串
def lengthOfLongestSubstring(s):
    if not s:
        return 0
    res = ''
    max_len = cur_len = 0
    for ch in s:
        if ch not in res:
            res += ch
            cur_len = len(res)
        else:
            res += ch
            res = res[res.index(ch)+1:]
        if cur_len > max_len:
            max_len = cur_len
    return max_len
# print(lengthOfLongestSubstring('abcdebadfghtrd'))

# 给定一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？
# 找出所有满足条件且不重复的三元组。
def checkThreeNumSumZero(nums):
    if not nums:
        return []
    res = []
    nums.sort()
    for i in range(len(nums)):
        if nums[i] > 0:  # 当前元素大于0，则后面必然不存在a+b+c=0的三元素
            return res
        # 排序后相邻两数如果相等，则跳出当前循环继续下一次循环，相同的数只需要计算一次
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i+1, len(nums)-1
        while left < right:
            tmp = nums[i] + nums[left] + nums[right]
            if tmp == 0:
                res.append([i, left, right])
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


# 将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的
def mergeList(l1, l2):
    head = Node(-1)
    cur_node = head
    while l1 or l2:
        if l1.data < l2.data:
            cur_node.next = l1
            l1 = l1.next
        else:
            cur_node.next = l2
            l2 = l2.next
        cur_node = cur_node.next
    if l1:
        cur_node.next = l1
    if l2:
        cur_node.next = l2
    return head.next


# 旋转数组中查找指定数字
def findTargetInRetateArray(nums, target):
    # 旋转数组是把有序数组翻转得到的，通常在有序数组中使用二分查找，这里部分有序数组同样可以使用二分查找
    if not nums:
        return None
    low, high = 0, len(nums)-1
    while low <= high:
        mid = low + (high - low) // 2  # 防止两个大数相加溢出问题
        # 如何确定当前数字属于前半子数组 or 后半子数组
        if nums[mid] == target:
            return mid
        # 本题较标准二分法多了判断mid归属于哪个子数组
        if nums[mid] < nums[high]:  # mid指向后半子数组
            if nums[mid] < target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1
        else:
            if nums[low] <= target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
# print(findTargetInRetateArray([6,7,1,2,3,4,5], 4))

# 旋转数组中最小数字
def findMinNumInRetateArray(nums):
    # 二分法
    low, high = 0, len(nums)-1
    # while nums[low] >= nums[high]:  # 终止条件
    while low <= high:  # 终止条件
        if high - low == 1:
            return nums[high]
        mid = low + (high - low) // 2
        if nums[low] == nums[mid] == nums[high]:
            # 顺序查找
            min_val = nums[low]
            for num in nums[low: high + 1]:
                if num < min_val:
                    min_val = num
            return min_val
        if nums[mid] >= nums[low]:  # mid指向前半子数组
            low = mid  # 这里不是mid+1是因为low要指定前子数组的最后一位
        else:
            high = mid
    return None
# print(findMinNumInRetateArray([6,7,1,2,3,4,5]))

# 所有和为s的连续序列，例如输入为9，输出为[[2, 3, 4], [4, 5]]
def find_continuous_sequence(tsum):
    # 双指针法，start=0,end=1, 若start+...+end等于tsum,则记录,start+1；若大于tsum，start+1；若小于tsum,则end+1
    start, end = 0, 1
    res = []
    while start < (tsum + 1) // 2:  # 只需要执行一半即可
        cur_sum = sum(range(start, end+1))
        if cur_sum == tsum:
            res.append(list(range(start, end+1)))
            start += 1
        elif cur_sum < tsum:
            end += 1
        else:
            start += 1
    return res
# print(find_continuous_sequence(15))

# 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的
def twoNumSum(nums, target):
    # 前后指针
    if not nums:
        return None
    low, high = 0, len(nums)-1
    while low < high:  # 终止条件: 两个指针不相遇
        if nums[low] + nums[high] == target:
            return [nums[low], nums[high]]
        elif nums[low] + nums[high] < target:
            low += 1
        else:
            high -= 1
    return None
# print(twoNumSum([1,2,3,4,5,6,7,8,9], 15))

# 数组中1个数字出现奇数次，其余数组出现偶数次，找出该数字
# 异或: 0 ^ x = x, x ^ x = 0
def findNumApperOddTimes(nums):
    res = 0
    for num in nums:
        res ^= num
    return res
# print(findNumApperOddTimes([1,2,3,4,1,2,3,4,3]))

# 升级：数组中有2个数字出现奇数次，其余出现偶数次，找出这2个数字
def findNumsApperOddTimes(nums):
    # step1: 异或
    res = 0
    for num in nums:
        res ^= num
    # step2: 在res中找最低位非0的位
    ix = 0
    if res & 0x01 == 0:
        res = res >> 1
        ix += 1

    # step3：根据ix位是否是1，分为2个子数组，每个子数组中有1个数字出现奇数次，其余出现偶数次
    res1, res2 = [], []
    for num in nums:
        if num & (0x01 << ix):
            res1.append(num)
        else:
            res2.append(num)
    # step4: 分别找到奇数次数字
    return [findNumApperOddTimes(res1), findNumApperOddTimes(res2)]
# print(findNumsApperOddTimes([1,2,3,4,2,3,6,6]))

# 在字符串中找出第一次出现1次的字符
def findStrFirstApperOnce(s):
    if not s:
        return None
    # 查找次数首选哈希标
    hash_table = [0] * 256
    for ch in s:
        hash_table[ord(ch)] += 1
    print(hash_table)
    for ch in s:
        if hash_table[ord(ch)] == 1:
            return ch
    return None
# print(findStrFirstApperOnce('abcdab'))

# 统计一个数字在排序数字中出现的次数
def get_first_num(nums, target):
    start, end = 0, len(nums)-1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            if mid == start or nums[mid-1] != target:
                return mid
            else:
                end = mid - 1
        elif nums[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    return -1

def get_last_num(nums, target):
    start, end = 0, len(nums) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            if mid == end or nums[mid + 1] != target:
                return mid
            else:
                start = mid + 1
        elif nums[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    return -1

def getNumOfK(nums, target):
    # 本体属于二分法的变形，需要找到第一个和最后一个位置
    if not nums:
        return None
    first_ix = get_first_num(nums, target)
    last_ix = get_last_num(nums, target)
    if first_ix == -1 and last_ix == -1:
        return -1
    return last_ix - first_ix + 1
# print(getNumOfK([1,2,3,3,3,3,3,3,4,5], 3))


# 树的深度 = max(左子树深度，右子树深度) + 1
def treeDepth(root):
    if not root:
        return 0
    left_depth = treeDepth(root.left)
    right_depth = treeDepth(root.right)
    return max(left_depth, right_depth) + 1

# 是否是平衡树
def isBalanceTree(root):
    if not root:
        return True

    def isBalanceTreeHelper(root):
        # 从底部开始
        if not root:
            return 0
        left_depth = isBalanceTreeHelper(root.left)
        right_depth = isBalanceTreeHelper(root.right)
        if left_depth == -1 or right_depth == -1:  # 左右子树有1个不平衡，则整体不平衡
            return -1
        return -1 if abs(left_depth - right_depth) > 1 else 0  # 高度差大于1则不平衡

    return isBalanceTreeHelper(root) == -1

# 给定5张牌是否是顺子，0是大王
def isStraight(nums):
    # 若是顺子，则除0外其他牌的差不能大与4
    if not nums or len(nums) < 5:
        return False
    hash_table = []
    for num in nums:
        if num == 0:
            continue
        if num in hash_table:
            return False
        hash_table.append(num)
    return True if (max(hash_table) - min(hash_table)) <=4 else False
# print(isStraight([1,2,3,4,0]))


# 圆圈中剩余的最后数字
def remainNum(n, m):
    peoples = list(range(1, n+1))
    ix = 0
    while len(peoples) > 1:
        ix = (ix + m - 1) % len(peoples)
        del peoples[ix]
    return peoples[0]
# print(remainNum(4, 3))

# 左旋转字符串:字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”
def leftRetateStr(s, n):
    # 方法1
    n = n % len(s)
    return s[n:] + s[:n]

    # # 方法2：先翻转前n个字符，后x个字符，整体翻转
    # n = n % len(s)
    # s = s[:n][::-1] + s[n:][::-1]
    # return s[::-1]
# print(leftRetateStr('abcXYZdef', 12))

# 反转单词顺序
def reverseSent(s):
    if not s:
        return None
    # 先反转整个句子，然后翻转单词

    def reverse(s, start, end):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1
        return ''.join(s)

    s = list(s)
    s = reverse(s, 0, len(s)-1)
    words = s.split(' ')
    res = []
    for word in words:
        res.append(reverse(list(word), 0, len(word)-1))
    return ' '.join(res)
# print(reverseSent('i am a student'))

# 寻找两个有序数组的中位数
# 盛最多水的容器
def maxArea(heights):
    # 实际上就是前后指针，求最大面积
    if not heights:
        return 0
    max_area = 0
    i, j = 0, len(heights)-1
    while i != j:
        area = min(heights[i], heights[j]) * (j - i)
        if area > max_area:
            max_area = area
        # 移动指针，规则是移动小的指针
        if heights[i] < heights[j]:
            i += 1
        else:
            j -= 1
    return max_area
# print(maxArea([1,8,6,2,5,4,8,3,7]))

# 最大子序列和
def maxSubSum(nums):
    if not nums:
        return 0
    cur_sum = max_sum = float('-inf')
    for num in nums:
        if cur_sum <= 0:  # 若当前和小于等于0则加num必然小于num，所以直接令当前最大和为num
            cur_sum = num
        else:
            cur_sum += num
        if cur_sum > max_sum:
            max_sum = cur_sum
    return max_sum
# print(maxSubSum([-2,1,-3,4,-1,2,1,-5,4]))

# 子集：给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
def getSub(nums):
    # 从[]开始，从前往后遍历nums,每次把当前已有子集和遍历到的数字组合为新的子集
    res = [[]]
    for num in nums:
        new = [tmp+[num] for tmp in res]
        res += new
    return res
# print(getSub([1,2,3]))

# 对称的二叉树
def mirrorTree(tree):
    if not tree:
        return False

    def mirritTreeHelper(left_tree, right_tree):
        # 检查左子树和右子树是否对称
        if left_tree is None and right_tree is None:
            return True
        if left_tree is None or right_tree is None:
            return False
        if left_tree.data != right_tree.data:
            return False
        return mirritTreeHelper(left_tree.left, right_tree.right) \
               and mirritTreeHelper(left_tree.right, right_tree.left)

    return mirritTreeHelper(tree.left, tree.right)


def maxDepth(root):
    if not root:
        return 0
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    return max(left_depth, right_depth)+1


# 路径和：判断是否存在路径，使得路径值和等于target
def hasPath(root, target):
    # 路径：从根节点到叶子节点，先序遍历
    if root is None:
        return False
    # 叶子节点且值相等
    if root.left is None and root.right is None and target == root.val:
        return True
    return hasPath(root.left, target-root.val) or hasPath(root.right, target-root.val)


# 获得所有路径和为target的路径
def getPash(root, target):
    # 先序遍历，需要在不符合时弹出节点
    res = []
    if root is None:
        return res

    def helper(node, tmp, sum):
        if node is None:
            return
        if node.left is None and node.right is None and sum == node.val:
            tmp.append(node)
            res.append(tmp)
        helper(node.left, tmp+[node.val], sum-node.val)
        help(node.right, tmp+[node.val], sum-node.val)

    helper(root, [], target)


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

# print(minEditDist('batyu', 'beatyu'))


# 判断链表中是否有环
# 快慢指针，p1每次走1步，p2每次走两步，若相遇则说明有环
def hasCirle(head):
    if head is None or head.next is None:
        return False
    p1 = head
    p2 = head.next
    while p2 and p2.next:
        p1 = p1.next
        p2 = p2.next.next
        if p1 == p2:
            return True
    return False


# 翻转链表
# 三指针法
def reverseLink(head):
    if head is None or head.next is None:
        return head
    pre_node = None
    cur_node = head
    while cur_node:
        next_node = cur_node.next
        cur_node.next = pre_node

        pre_node = cur_node
        cur_node = next_node
    return pre_node


# 翻转二叉树
# 左右孩子交换
def reverseTree(root):
    if not root:
        return None
    root.left, root.right = root.right, root.left
    reverseTree(root.left)
    reverseTree(root.right)
    return root

# 回文数，回文串：正读和反读相同
def isisPalindromeNum(key):
    if isinstance(key, str):
        key = key.lower()
    return key == key[::-1]


# 移动0，把0元素移动到末尾，非0保持不变
def removeZeros(nums):
    # 方法1：辅助列表
    def helper1(nums):
        res1, res2 = [], []
        for num in nums:
            if num == 0:
                res1.append(num)
            else:
                res2.append(num)
        return res2+res1
    helper1(nums)

    def helper2(nums):
        # 原址排序，双指针法
        j = 0  # j指向0值，默认开头便是0
        size = len(nums)
        for i in range(size):  # i遍历数据，指定非0值
            if nums[i] != 0:
                nums[j] = nums[i]  # 把j位置置为非零值
                if i != j:  # i!=j,表示不同位置,把i位置置为0
                    nums[i] = 0
                j += 1
    helper2(nums)


# 找到所有数组中消失的数字
def findDisappearedNumbers(nums, n):
    if not nums:
        return []
    return set(range(1, n+1)) - set(nums)
# print(findDisappearedNumbers([1,3,3,5,5], 5))


# 把二叉搜索树转换为累加树
pre_node_val = 0

# 合并二叉树
def mergeTree(root1, root2):
    # 先序遍历
    # 合并规则：
    #   case1: 两棵树的当前节点都不为空，则相加值
    #   case2：若有1棵树的当前节点为空，则返回不为空的树的节点
    #   case3：若都为空，则直接返回空
    if not root1 or not root2:
        return root1 or root2

    root1.val += root2.val
    root1.left = mergeTree(root1.left, root2.left)
    root1.right = mergeTree(root1.right, root2.right)
    return root1

# 数字二进制中1的个数
def oneCounts(num):
    cnt = 0
    while num:
        num &= (num-1)
        cnt += 1
    return cnt
# print(oneCounts(3))

# 汉明距离：两个整数之间的汉明距离指的是这两个数字对应二进制位不同的位置的数目
def hanmingDict(num1, num2):
    # 两数异或后求结果二进制中1的个数
    res = num1 ^ num2
    return oneCounts(res)
# print(hanmingDict(3, 5))

# 最小栈
# O(1)时间内获取当前栈中最小值
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # 记录当前栈最小值

    def push(self, item):
        # 入栈
        self.stack.append(item)
        if not self.min_stack:
            self.min_stack.append(item)
            return
        # 重新入栈当前最小值
        if item < self.min_stack[-1]:
            self.min_stack.append(item)
        else:
            self.min_stack.append(self.min_stack[-1])

    def pop(self):
        # 出栈
        if not self.stack:
            return None
        self.min_stack.pop()
        self.stack.pop()

    def top(self):
        if not self.stack:
            return None
        return self.stack[-1]

    def get_min(self):
        if not self.min_stack:
            return None
        return self.min_stack[-1]

# min_stack = MinStack()
# min_stack.push(5)
# min_stack.push(3)
# min_stack.push(10)
# min_stack.push(1)
# print(min_stack.top())
# print(min_stack.get_min())
# min_stack.pop()
# print(min_stack.top())
# print(min_stack.get_min())

# 多次元素：次数超过一半的元素
#   方法1：快排的partition()
#   方法2：利用数字超过一半性质，同一元素次数加1，非同一元素次数减1，最后次数大于1的元素即为所求
#   方法3：字典
def moreHalfNum(nums):
    if not nums:
        return None
    def helper2():
        num = nums[0]
        time = 1
        for i in range(1, len(nums)):
            if num == nums[i]:  # 同一元素次数加1，非同一元素次数减1
                time += 1
            else:
                time -= 1
            if time <= 0:  # time<=0表示当前不同元素较多
                num = nums[i]
        return num
    # return helper2()

    def helper3():
        num_cnt = {}
        for num in nums:
            if num not in num_cnt:
                num_cnt[num] = 1
            else:
                num_cnt[num] += 1
            if num_cnt[num] > len(nums)//2:
                return num
        return None
    return helper3()
# print(moreHalfNum([1,2,2,2,3,4, 2]))


# 相交链表，求两个链表的第一个公共节点
def commonNode(head1, head2):
    # 链表有1个为空，或者全都为空，则没有公共节点
    if not head1 or not head2:
        return None

    def helper1():
        # 双指针问题
        # 长的链表先走abs(len(head1) - len(head2))步，然后一起走，直到第一个相遇的节点
        node1, node2 = head1, head2
        len1 = len2 = 0
        while node1:
            len1 += 1
            node1 = node1.next
        while node2:
            len2 += 1
            node2 = node2.next
        pLong, pShort = head1, head2
        if len1 < len2:
            pLong = head2
            pShort = head1
        for i in range(abs(len1-len2)):
            pLong = pLong.next
        while pLong and pShort:
            if pLong == pShort:
                return pLong
            pLong = pLong.next
            pShort = pShort.next
        return None
    # return helper1()

    def helper2():
        # 简单方法：你走过我的路，我走过你的路，最终我们相遇了
        p1, p2 = head1, head2
        while p1 or p2:  # 终止条件：两个链表都到达末尾
            if p1 == p2:
                return p1
            p1 = p1.next
            p2 = p2.next
            if p1 is None:
                p1 = head2
            if p2 is None:
                p2 = head1
    return helper1()

# 合并两个有序数组，要求在nums1上合并
def mergeTwoNums(nums1, m, nums2, n):
    # 思路: 从尾到头
    # 参数检查
    if not nums1 or not nums2 or len(nums2) < n or len(nums1) < m+n:
        return []
    while m > 0 or n > 0:  # 终止条件：有一个数组遍历到开始处
        if nums1[m] > nums2[n]:
            nums1[m+n-1] = nums1[m]
            m -= 1
        else:
            nums1[m+n-1] = nums2[n]
            n -= 1
    # 若nums2还有元素，即nums2中元素较小
    if n > 0:
        nums1[:n] = nums2[:n]
    return nums1

# 两个数组的交集
def intersection(num1, num2):
    # 方法1: set(num1) & set(num2)
    # 方法2: [x for x in num1 if x in num2]
    pass

# 两整数之和
def sumTwoNum(num1, num2):
    # 不考虑进位: 两个数字二进制相加，1+1=0，0+1=1，实际上就是异或操作
    # 考虑进位: 1+1有进位，0+1无进位，实际上就是位于操作后左移1位
    while num2:  # 终止条件：无进位为止
        _sum = num1 ^ num2
        carry = (num1 & num2) << 1
        num1 = _sum
        num2 = carry
    return num1
# print(sumTwoNum(5, 7))

# 字符串中第一个唯一字符
# 给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1
def findStrApperenceOnce(s):
    # 字符次数，首选哈希标
    ch_cnt = {}
    for ch in s:
        if ch in ch_cnt:
            ch_cnt[ch] += 1
        else:
            ch_cnt[ch] = 0

    for i in range(len(s)):
        if ch_cnt[s[i]] == 1:
            return i
    return -1

# 左叶子节点和
# 难点: 如何判断节点是左叶子节点
# 叶子节点: node.left is None and node.right is None
# 左叶子节点: node.left is not None and node.left.left is None and node.left.right is None -> node.left 是node的左叶子节点
def leftLeafNodeSum(root):
    # 先序遍历
    if not root:
        return 0
    if root.left and root.left.left is None and root.left.right is None:  # 左叶子节点
        return root.left.val + leftLeafNodeSum(root.right)  # 因为root.left已经是左叶子节点，所以无需在递归left孩子
    return leftLeafNodeSum(root.left) + leftLeafNodeSum(root.right)  # 递归遍历左右子树


# 题目描述：给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数
def twoSum(nums, target):
    # 前后指针问题
    if not nums:
        return None
    start, end = 0, len(nums)-1
    while start < end:
        cSum = nums[start] + nums[end]
        if cSum == target:
            return [start, end]
        elif cSum > target:
            end -= 1
        else:
            start += 1
# print(twoSum([2,7,9,11], 9))

# 删除排序数组中的重复项，对于重复项保留1个[1,2,2,3] -> [1,2,3]
def deleteDuplicates(nums):
    # 方法1：return list(set(nums))
    if not nums:
        return nums
    for i in range(len(nums)):
        if nums[i] == nums[i+1] and i < len(nums)-1:  # 若当前值等于下一个值，删除下一个值
            nums.pop(i+1)
    return len(nums)

# # 给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一
def addOne(nums):
    if not nums:
        return nums
    res = 0
    for num in nums:
        res = 10 * res + num
    print(res)
    res += 1
    ret = []
    while res:
        ret.insert(0, res % 10)
        res = res // 10
    return ret
# print(addOne([1,2,9]))

# 反转整数
def reverseNum(num):
    return int(str(num)[::-1])
# print(reverseNum(123))


# 和最少为k的最短子数组
# 返回 A 的最短的非空连续子数组的长度，该子数组的和至少为 K 。
def findMinLen(nums, k):
    # 找出所有数组
    if not nums:
        return nums
    ret = []
    nums_len = len(nums)
    for i in range(nums_len):  # 遍历start
        _sum = nums[i]
        for j in range(i+1, nums_len):  # 遍历end
            _sum += nums[j]
            if _sum >= k:
                ret.append(j-i+1)  # 找到后跳出，因为往后找长度只会更大
                break
    return min(ret)
# print(findMinLen([2,-1,2], 3))

# 设计循环队列
# 循环队列有头尾指针，主要在于如何判断是否空队列，满队列
class Solution:
    def __init__(self, k):
        self.data = [None] * (k+1)    # 数据
        self.front = 0    # 头指针
        self.rear = 0     # 尾指针

    def is_empty(self):
        # 头尾指针相等，则空队列
        return self.front == self.rear

    def is_full(self):
        # 头尾指针相差1，则满队列
        return (self.rear + 1) % len(self.data) == self.front

# 删除链表的倒数第N个节点
# 给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点
def deleteNode(head, n):
    # 方法1: 首先遍历列表统计列表长度len；然后从开走len-n-1步，该节点是待删除节点的前一节点，删除即可
    # 方法2：快慢指针，快指针先走n步，然后两个指针一起走，快指针走到最后一个节点时，满指针是待删除节点的前一节点
    def helper1():
        if not head or n <= 0:
            return None
        node_cnt = 0
        node = head
        while node:
            node_cnt += 1
            node = node.next

        if node_cnt < n:
            print('error')
            return None
        node = head
        if n == node_cnt:  # 表示删除第一个节点
            return head.next
        for i in range(node_cnt-n-1):
            node = node.next
        node.next = node.next.next
        return head

    def helper2():
        if not head or n <= 0:
            return None
        fast = slow = head
        head_len = 0
        for i in range(n):
            if fast:
                fast = fast.next
                head_len = i
        if head_len < n:  # 链表长度小于n
            return None
        if fast is None:  # n等于链表长度，返回头节点
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head

# 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次
def headDuplicate(head):
    if not head or not head.next:
        return head
    slow, fast = head, head.next
    while slow:  # 终止条件: 遍历到最后
        # 循环遍历，直到非重复为止
        while fast and fast.val == slow.val:
            fast = fast.next
        # 跳过重复节点
        slow.next = fast
        slow = fast
        if fast.next:  # fast不到最后一个节点
            fast = fast.next
    return head


# 环的入口节点
def detectCycle(head):
    # step1：检查是否有环
    # step2：确定环长度
    # step3：找入口节点
    if not head or not head.next:
        return None

    node = hasCycle(head)
    if node is None:
        return None

    node_nums = cycle_node_nums(node)
    return first_detect_cycle(head, node_nums)

def hasCycle(head):
    # 是否存在环
    # 方法: 快慢指针，快指针每次走两步，满指针每次走一步，若相遇则必然存在环
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return slow
    return None

def cycle_node_nums(head):
    # 环中节点个数，遍历即可
    node = head
    cnt = 1
    while node.next != head:
        node = node.next
        cnt += 1
    return cnt

def first_detect_cycle(head, nums):
    # 找环的入口节点
    # 快慢指针，快指针先走nums步，然后一起走，相遇点即为所求点
    fast, slow = head, head
    for i in range(nums):
        fast = fast.next
    while fast != slow:
        fast = fast.next
        slow = slow.next
    return fast

# 移除链表元素：删除链表中等于给定值 val 的所有节点
def removeNode(head, key):
    if not head or head.val == key:
        return None
    # 构造一个头节点
    firstnode = Node(-1)
    firstnode.next = head
    pre, cur = firstnode, head
    while cur:  # 终止条件: 遍历到最后节点
        if cur.val == key:  # 重复，跳过cur节点
            pre.next = cur.next
            cur = cur.next
        else:
            pre = pre.next
            cur = cur.next
    return firstnode.next

# 反转链表
def reverseList(head):
    # 三指针法
    if not head or not head.next:
        return head
    new_head = Node(-1)
    new_head.next = head
    pre = new_head
    cur = head
    while cur:
        pnext = cur.next
        cur.next = pre
        pre = cur
        cur = pnext
    return new_head.next

# 回文链表
def isPalindromeLink(head):
    # 遍历链表列表记录节点值，问题转化为列表是否是回文
    if not head or not head.next:
        return False
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res == res[::-1]

# 相同的树: 结构相同，节点值相同
def isSameTree(root1, root2):
    if not root1 and not root2:  # 两棵树都遍历完成，表示是相同的树
        return True
    if not root1 or not root2:   # 只有1棵树遍历完成
        return False
    if root1.val != root2.val:   # 值不同，不是相同的树
        return False
    left = isSameTree(root1.left, root2.left)
    right = isSameTree(root1.right, root2.right)
    return left and right

# 对称的树: 树的左孩子等于右孩子
def isMirrorTree(root):
    if not root:
        return True
    return isMirrorTreeHelper(root.left, root.right)

def isMirrorTreeHelper(left_root, right_root):
    if not left_root and not right_root:  # 遍历完成，是对称树
        return True
    if not left_root or not right_root:
        return False
    if left_root.val != right_root.val:
        return False
    left = isMirrorTreeHelper(left_root.left, right_root.right)
    right = isMirrorTreeHelper(left_root.right, right_root.left)
    return left and right

# 将有序数组转换为高度平衡的二叉搜索树
# 思路：二叉搜索树的中序遍历是有序的，因此可以把有序数组认为是二叉搜索树的中序遍历得到的
#      首先要构建根节点，方法有很多，第一个认为是根节点或者任1个是根节点(只要满足根节点前面的值小于根节点值，后面的大于根节点值)
#           但是如果要满足高度平衡，那么最好认为中间的值是根节点
#      然后，递归构建左右子树
def sortArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortArrayToBST(nums[:root])
    root.right = sortArrayToBST(nums[root+1:])
    return root

# 二叉树的最小深度
# 注意：本题和二叉树最大深度不同，不能直接min(left, right)
# 最小深度是从根节点到最近叶子节点的最短路径上的节点数量
# 思路：层序遍历，遇到叶子节点则返回
def min_depth(root):
    if not root:
        return 0
    queue = [root]
    depth = 1
    while queue:
        tmp = []
        for node in queue:  # 每次必须把当前层所有节点的所有子节点添加到列表中，按层往下
            if node.left is None and node.right is None:
                return depth
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
        queue = tmp
        depth += 1
    return depth

# 最长公共前缀
def maxLenCommonPrefix(strs):
    # 思路: 先找出最短的字符串s，然后逐个匹配是否是公共前缀，若到i位置匹配失败，则s[:i]即为最长公共前缀
    if not strs:
        return 0
    min_str = min(strs, key=len)  # 找最短字符串
    for ix, ch in enumerate(min_str):  # 遍历每个字符
        for other in strs:             # 匹配其他字符串
            if other[ix] != ch:
                return min_str[:ix]
    return min_str
# print(maxLenCommonPrefix(['abc', 'abcd', 'ab']))

# 字符串匹配,haystack中是否有needle串
def strStr(haystack, needle):
    # return haystack.index(needle)
    # return haystack.find(needle)

    # 朴素匹配法
    def strStrHelper(haystack, needle):
        i = j = 0
        while i < len(haystack) and j < len(needle):  # 终止条件: 两个串有一个遍历完
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                i = i - j + 1  # 匹配失败，i回到本次匹配初始位置的下一位置，j为0
                j = 0
        if j == len(needle):   # 匹配成功,返回在主串中的索引
            return i - j
        return -1

# TODO：滑动窗口最大值

def partition(nums, start, end):
    # 把大于key的放到后面，小于key的放到前面
    key = nums[start]
    while start < end and nums[end] >= key:
        end -= 1
    nums[start], nums[end] = nums[end], nums[start]
    while start < end and nums[start] <= key:
        start += 1
    nums[start], nums[end] = nums[end], nums[start]
    return start


# 在未排序数组中找第二大数
def findSecondMaxValue(nums, k):
    # 方法1：排序后取num[1]  O(NlogN)
    # 方法2：Partition()
    if not nums:
        return None

    def helper_part(nums, k):
        start = 0
        end = len(nums)-1
        index = partition(nums, start, end)
        while index != k - 1:
            if index < k - 1:  # 在左边递归查找
                index = partition(nums, index + 1, end)
            if index > k - 1:  # 在右边递归查找
                index = partition(nums, 0, index - 1)
        return nums[index]
    # return helper_part(nums, k)

    def helper2(nums):
        # 记录最大和次大数，遍历更新
        max_num = second_num = float('-inf')
        for num in nums:
            if num > max_num:  # 若当前值大于最大值，则更新次大值和最大值
                second_num = max_num
                max_num = num
            elif num > second_num:  # 若次大值 < 当前值 < 最大值，则只更新次大值
                second_num = num
        return max_num, second_num
    return helper2(nums)

# print(findSecondMaxValue([10, 20, 5, 6], 2))

# 最大子序列和
# 在给定序列中查找具有最大和的子序列
def maxSumArray(nums):
    # 方法1
    def helper1(nums):
        # 遍历序列，若当前和小于0，则当前和等于当前值，更新最大和
        cur_sum = max_sum = float('-inf')
        for num in nums:
            if cur_sum <= 0:
                cur_sum = num
            else:
                cur_sum += num
            if cur_sum > max_sum:
                max_sum = cur_sum
        return max_sum
    print('helper1: ', helper1(nums))

    def helper2(nums):
        # 贪心算法，每次都做出在当前最好的
        cur_sum = max_sum = nums[0]
        for num in nums[1:]:
            cur_sum = max(cur_sum, cur_sum+num)
            max_sum = max(max_sum, cur_sum)
            print(num, cur_sum, max_sum)
        return max_sum
    # print('helper2: ', helper2(nums))

    def helper3(nums):
        # 动态规划
        # dp[i]表示遍历到第i个数时当前最大和
        # 状态转移方程: dp[i] = dp[i-1] + num if dp[i-1]>0 else num
        dp = [float('-inf')] * len(nums)
        for i, num in enumerate(nums[1:]):
            dp[i] = dp[i-1] + num if dp[i-1] > 0 else num
        print(dp)
        return dp[-1]
    # print('helper3: ', helper3(nums))
# print(maxSumArray([-2,1,-3,4,-1,2,1,-5,4]))


# 判断子序列：给定字符串 s 和 t ，判断 s 是否为 t 的子序列。
# 字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。
def isSubStr(s, t):
    # 注意这里和字符串匹配算法不同
    # 思路：只需要移动主串，匹配串不移动
    i = j = 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
            j += 1
        else:  # 只移动主串s
            i += 1
    if j == len(t):  # 匹配串t匹配完成
        return True
    return False
# print(isSubStr("abcde", "ace"))

# 字符串相加 '123' + '111' = '234'
def str_add(str1, str2):
    # 转换为数字后相加，在转换为字符串
    # return str(int(str1) + int(str2))

    if not str1 or not str2:
        return str1 or str2
    sum1 = sum2 = 0
    for ch in str1:
        sum1 = 10 * sum1 + ord(ch) - ord('0')
    for ch in str2:
        sum2 = 10 * sum2 + ord(ch) - ord('0')
    return str(sum1 + sum2)
# print(str_add('111', '222'))

# 字符串中的单词数
# 单词以空格间隔
# print(len(" a b   c ".strip().split()))

# 偶数在奇数前
# sorted(nums, key=lambda x: x & 1)
# 奇数在偶数前
# sorted(nums, key=lambda x: x & 1, reverse=True)
def sortArrayByParity(lst):
    # 方法1：辅助列表，分别保存奇数、偶数
    # 方法2：sorted()
    # 方法3：双指针法
    length = len(lst)
    p_begin = 0
    p_end = length-1
    while p_end >= p_begin:
        """
        p_end指向最后一个奇数
        p_begin指向第一个偶数
        """
        while p_end >= p_begin and not lst[p_end] & 1:
            p_end -= 1
        while p_end >= p_begin and lst[p_begin] & 1:
            p_begin += 1
        if p_end >= p_begin:
            lst[p_end], lst[p_begin] = lst[p_begin], lst[p_end]
    return lst
# print(sortArrayByParity([2,4,3,1]))

# 两棵树的最低公共祖先
# case1: 若树有指向父节点的指针，则该问题转换为"两个链表求第一个公共节点"
# case2: 若树没有指针父节点的指针，但是树为二叉搜索树
#   根据二叉搜索树的性质：左子树节点值 < 根节点值 < 右子树节点值，先序遍历树
#   1、若当前节点值大于给定两个节点的值，则最低公共祖先在当前节点的左子树
#   2、若当前节点值小于给定两个节点的值，则最低公共祖先在当前节点的右子树
#   3、若当前节点值介于给定两个节点的值的中间，则该节点即为所求
# case3: 没有指向父节点的指针，普通树
#   需要前序遍历树，保存从根节点到指定节点的路径，最后在路径中找最低公共祖先节点

# BST树实现
def findCommonNodeHelper(root, node1, node2):
    if not root:
        return
    if root.val < node1.val and root.val < node2.val:    # 递归右子树
        findCommonNodeHelper(root.right, node1, node2)
    elif root.val > node1.val and root.val > node2.val:  # 递归左子树
        findCommonNodeHelper(root.left, node1, node2)
    else:
        return root

def findCommonNode_bst(root, node1, node2):
    if not root or not node1 or not node2:
        return None
    return findCommonNodeHelper(root, node1, node2)

# 普通树实现：需要前序遍历树，保存从根节点到指定节点的路径，最后在路径中找最低公共祖先节点
def get_path(root, node, path):
    if not root:
        return
    if path and path[-1] == node:  # 找到目标node
        return

    path.append(root)
    get_path(root.left, node, path)
    get_path(root.right, node, path)
    if path and path[-1] != node:  # 回溯
        path.pop()

def findCommonNode_path(root, node1, node2):
    if not root or not node1 or not node2:
        return None

    path1, path2 = [], []
    get_path(root, node1, path1)
    get_path(root, node2, path2)

    # 从前往后，在2个路径中查找最后的相等节点，即为最低公共祖先节点
    res = None
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            res = path1[i]
    return res