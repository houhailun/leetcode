#!/usr/bin.env python
# -*- encoding:utf-8 -*-

"""
在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
示例:
现有矩阵 matrix 如下：
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。
给定 target = 20，返回 false。
"""


class Solution(object):
    def findNumberIn2DArray(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        # 异常检查
        if not matrix:
            return False

        # 方法1：暴力法 O(n*n)
        # return self.findNumberIn2DArray_core1(matrix, target)

        # 方法2：利用数组性质：从左到右递增，从上到下递增，因此可以从右上角或者左下角等特殊位置开始判断
        # 本题选择右上角
        # 若当前元素小于目标值，则往下查找；若大于，则往左查找
        return self.findNumberIn2DArray_core2(matrix, target)

    def findNumberIn2DArray_core1(self, matrix, target):
        rows = len(matrix)
        columns = len(matrix[0])
        for row in range(rows):
            for col in range(columns):
                if target == matrix[row][col]:
                    return True
        return False

    def findNumberIn2DArray_core2(self, matrix, target):
        rows = len(matrix)
        columns = len(matrix[0])
        row = 0
        col = columns-1
        while row < rows and col >= 0:
            if matrix[row][col] > target:
                col -= 1
            elif matrix[row][col] < target:
                row += 1
            else:
                return True

        return False

if __name__ == "__main__":
    obj = Solution()

    matrix = [[1,   4,  7, 11, 15],
              [2,   5,  8, 12, 19],
              [3,   6,  9, 16, 22],
              [10, 13, 14, 17, 24],
              [18, 21, 23, 26, 30]]
    target = 24

    print(obj.findNumberIn2DArray(matrix, target))