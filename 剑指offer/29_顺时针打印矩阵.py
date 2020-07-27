#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        # 难点在于如何判断打印条件和下标更新
        if not matrix:
            return matrix
        result = []
        rows, columns = len(matrix), len(matrix[0])  # 行数，列数
        left, right, top, bottom = 0, columns-1, 0, rows-1  # 左，右，上，下四个角
        while left <= right and top <= bottom:  # 循环终止条件

            # 打印顺时针的第一步，行不变，列变)
            for i in range(left, right + 1):
                result.append(matrix[top][i])
            # 打印顺时针的第二步，条件为至少2行, 行变列不变
            if top < bottom:
                for i in range(top + 1, bottom + 1):
                    result.append(matrix[i][right])
            # 打印顺时针的第三步，条件为至少2列，行不变，列表
            if left < right and top < bottom:
                for i in range(right - 1, left - 1, -1):
                    result.append(matrix[bottom][i])
            # 打印顺时针的第四步，条件为至少3行，行变，列不变
            if top + 1 < bottom and left < right:
                for i in range(bottom - 1, top, -1):
                    result.append(matrix[i][left])
            left += 1
            right -= 1
            top += 1
            bottom -= 1
        return result


obj = Solution()
print(obj.spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]]))
