#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/3 17:03
# Author: Hou hailun

# 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。
# 如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。
# [["a","b","c","e"],
# ["s","f","c","s"],
# ["a","d","e","e"]]
#
# 但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。
# 示例 1：
# 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
# 输出：true
# 示例 2：
# 输入：board = [["a","b"],["c","d"]], word = "abcd"
# 输出：false

# 回溯法：
# 需要同样shape的数组visited来记录某一格子是否已经访问，防止重复访问


class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        # 健壮性检查
        if not board or not word:
            return False

        rows = len(board)
        columns = len(board[0])
        # visited = [[0] * columns] * rows
        visited = [[0 for _ in range(columns)] for _ in range(rows)]

        # 矩阵中的所有点都要作为起始点
        for i in range(rows):
            for j in range(columns):
                if self.exist_core(board, rows, columns, i, j, word, visited):
                    return True
        return False

    def exist_core(self, board, rows, columns, i, j, word, visited):
        # 遍历完成
        if not word:
            return True

        # 异常情况
        if i < 0 or i >= rows or j < 0 or j >= columns or board[i][j] != word[0] or visited[i][j] == 1:
            return False

        # 标记已访问
        visited[i][j] = 1

        # 递归 上下左右 方向，如果有一个方向可存在word路径，则返回
        # 不存在，则标记visited[i][j]未访问，便于下次访问
        if (self.exist_core(board, rows, columns, i + 1, j, word[1:], visited) or
                self.exist_core(board, rows, columns, i - 1, j, word[1:], visited) or
                self.exist_core(board, rows, columns, i, j - 1, word[1:], visited) or
                self.exist_core(board, rows, columns, i, j + 1, word[1:], visited)):
            return True

        # 回溯，将当前位置的布尔值标记为未访问
        visited[i][j] = 0
        return False




obj = Solution()
a = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
b = "ABCCED"
print(obj.exist(a, b))
