#!/usr/bin/env python
# -*- encoding:utf-8 -*-


def getDigitArr(nums):
    if not nums:
        return None
    res = []

    # 双指针
    # i指向连续数字字符的开始
    # j指向连续数字字符的结束
    length = len(nums)
    i = 0
    while i < length:
        if nums[i].isdigit():
            # print(i, j, res)
            # 继续往后找，直到非数字为止
            j = i  # 注意：这里必须j=i，不能等与i+1，如果最后一位是数字，那么j+1就会超出范围，下面判断条件j！=i+1就不满足
            while j < length:
                if nums[j].isdigit():
                    j += 1
                else:
                    break

            # j！=i,表示i后有连续出现数字  或者  最后一位是数字
            if j != i or j == length:
                res.append(nums[i:j])
            i = j + 1
        else:
            i += 1
    return res

print(getDigitArr("abc123cd4f687f4"))
