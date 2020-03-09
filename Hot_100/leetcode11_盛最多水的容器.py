#!/usr/bin/env python
# -*- coding:utf-8 -*-



#垂直的两条线段将会与坐标轴构成一个矩形区域，较短线段的长度将会作为矩形区域的宽度，两线间距将会作为矩形区域的长度，而我们必须最大化该矩形区域的面积。


class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        # 暴力法,考虑所有的组合
        # 时间复杂度: O(N*N)
        # maxarea = 0
        # for i in range(len(height)):
        #     for j in range(i+1, len(height)):
        #         maxarea = max(maxarea, min(height[i], height[j]) * (j-i))
        # return maxarea

        # 方法2:双指针法
        # 思路:  设置双指针 ii,jj 分别位于容器壁两端，根据规则移动指针（后续说明），并且更新面积最大值 res，直到 i == j 时返回 res
        # 指针移动规则: 每次移动短板
        maxarea = 0
        i, j = 0, len(height)-1
        while i != j:
            cur_area = min(height[i], height[j]) * (j - i)
            if cur_area > maxarea:
                maxarea = cur_area

            if height[i] < height[j]:
                i += 1
            else:
                j -= 1

        return maxarea


if __name__ == "__main__":
    obj = Solution()

    res = [1,8,6,2,5,4,8,3,7]
    print(obj.maxArea(res))


