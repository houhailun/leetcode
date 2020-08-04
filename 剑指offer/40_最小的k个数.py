#!/usr/bin/env python
# -*- encoding:utf-8 -*-


class Solution:
    def getLeastNumbers(self, arr, k):
        def partition(arr, start, end):
            key = arr[start]
            while start < end:
                while start < end and arr[end] >= key:
                    end -= 1
                arr[start], arr[end] = arr[end], arr[start]
                while start < end and arr[start] <= key:
                    start += 1
                arr[start], arr[end] = arr[end], arr[start]
            return start

        if not arr:
            return arr
        if len(arr) < k:
            return []

        while True:
            ix = partition(arr, 0, len(arr) - 1)
            if ix == k:
                return arr[:k]
            elif ix < k:
                ix = partition(arr, ix+1, len(arr) - 1)
            else:
                ix = partition(arr, 0, ix-1)
        return

obj = Solution()
print(obj.getLeastNumbers([0,1,2,1], 1))