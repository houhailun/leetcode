#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/3/4 15:30
# Author: Hou hailun


class Solution(object):
    def __init__(self):  # 机器人可以倒回来，但不能重复计数。
        self.count = 0

    def movingCount(self, m, n, k):
        visited = [[1 for i in range(n)] for j in range(m)]
        self.findWay(visited, 0, 0, k)  # 从（0，0）开始走
        return self.count

    def findWay(self, visited, i, j, k):
        if i >= 0 and j >= 0 and i < len(visited) and j < len(visited[0]) \
                and sum(list(map(int, str(i)))) + sum(list(map(int, str(j)))) <= k and visited[i][j] == 1:
            visited[i][j] = 0
            self.count += 1
            self.findWay(visited,i-1,j,k)
            self.findWay(visited,i+1,j,k)
            self.findWay(visited,i,j-1,k)
            self.findWay(visited,i,j+1,k)


obj = Solution()
print(obj.movingCount(3, 2, 17))
