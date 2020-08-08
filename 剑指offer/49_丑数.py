#!/usr/bin/env python
# -*- encoding:utf-8 -*-

def nthUglyNumber(n):
    """
    :type n: int
    :rtype: int
    """
    # 方法1：逐个判断当前数字是否是抽数
    # 缺点n较大值，所需时间较长
    # num = 0
    # count = 0
    # while count < n:
    #     num += 1
    #     if self.isUgly(num):
    #         count += 1
    # return num

    # 方法2：利用抽数只能被2，3，5整除的性质，若一个数x是抽数，那么2X,3x，5x必然也是抽数,下一个抽数是这三个中取小
    #   假设当前有n-1个抽数，最后的丑数为M
    #   则必然有一个数字ix2使得在它之前的所有数乘以2都小于M，而T2×2的结果一定大于M，同样对于3，5也存在
    res = [1]
    min2 = min3 = min5 = 0
    cnt = 1
    while cnt < n:
        min_num = min(res[min2] * 2, res[min3] * 3, res[min5] * 5)
        res.append(min_num)

        # 更新min2，min3，min5
        while res[min2] * 2 <= min_num:
            min2 += 1
        while res[min3] * 3 <= min_num:
            min3 += 1
        while res[min5] * 5 <= min_num:
            min5 += 1
        cnt += 1
    return res[-1]

print(nthUglyNumber(10))