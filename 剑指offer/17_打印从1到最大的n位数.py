#!/usr/bin/env python
# -*- encoding:utf-8 -*-

# 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

class Solution(object):
    def __init__(self):
        self.result = []

    def printNumbers(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if n < 1:
            return None

        # 方法1
        # 使用python的字符
        # return [i for i in range(1, int('9' * n) + 1)]

        # 方法2：power实现, 其他语言存在大数问题
        # res = []
        # for i in range(1, 10**n):
        #     res.append(i)
        # return res
        # pythonic：[i for i in range(1, 10**n)]

        # 大数问题一般可以用字符列表来解决，这里不用字符串是是因为python中字符串不能被修改
        return self.printNumbersCore(n)

    def printNumbersCore(self, n):
        if n <= 0:
            return []
        number = ["0"] * n
        number[-1] = "1"
        for i in range(0, 10):
            number[0] = chr(ord("0") + i)  # ord 是将一个字符转换成 ASCII 码，chr 是将一个 ASCII 码转换成一个数字
            self.Print1ToMaxOfDigitsRecursively(number, n, 0)
        return (self.result[1:])

    def Print1ToMaxOfDigitsRecursively(self, number, length, index):
        if index == length - 1:
            self.PrintNumberNormal(number)
            self.result.append(int("".join(number)))
            return

        for i in range(10):
            number[index + 1] = chr(ord("0") + i)
            self.Print1ToMaxOfDigitsRecursively(number, length, index + 1)

    def PrintNumberNormal(self, number):
        number = int("".join(number))
        if number != 0:
            print('PrintNumberNormal():', number)



obj = Solution()
print(obj.printNumbers(1))