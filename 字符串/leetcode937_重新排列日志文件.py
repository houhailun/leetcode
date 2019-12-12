#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/20 14:08
# Author: Hou hailun

"""
你有一个日志数组 logs。每条日志都是以空格分隔的字串。
对于每条日志，其第一个字为字母数字标识符。然后，要么：
    标识符后面的每个字将仅由小写字母组成，或；
    标识符后面的每个字将仅由数字组成。
我们将这两种日志分别称为字母日志和数字日志。保证每个日志在其标识符后面至少有一个字。
将日志重新排序，使得所有字母日志都排在数字日志之前。字母日志按内容字母顺序排序，忽略标识符；在内容相同时，按标识符排序。数字日志应该按原来的顺序排列。
返回日志的最终顺序。
示例 ：
输入：["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
输出：["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]
"""
import functools


class Solution(object):
    def reorderLogFiles(self, logs):
        """
        :type logs: List[str]
        :rtype: List[str]
        """
        # 1、把原日志列表一拆为二，拆分标准为标识符后面是数字还是字母
        # 2、对数字日志排序: 按原来顺序排序，即相对顺序保持不变
        # 3、对字母日志排序：内容不同按照字母顺序；内容相同按标识符顺序
        # digit_logs = []
        # alpha_logs = []
        # for log in logs:
        #     if log.split()[1].isdigit():
        #         digit_logs.append(log)
        #     else:
        #         alpha_logs.append(log)
        #
        # def cmp_func(x, y):
        #     x, y = x.split(), y.split()
        #     x_ix, y_ix = x[0], y[0]
        #     x, y = x[1:], y[1:]
        #     if x < y:  # -1表示x在y前面
        #         return -1
        #     elif x > y:
        #         return 1
        #
        #     if x_ix < y_ix:
        #         return -1
        #     elif x_ix > y_ix:
        #         return 1
        #     return 0
        #
        # ret = sorted(alpha_logs, key=functools.cmp_to_key(cmp_func))
        # ret.extend(digit_logs)
        # return ret

        # py2和py3的区别
        # py2 sorted(arr, cmp=func) 可以直接使用参数cmp指定自定义排序函数
        # py3 sorted(arr) 中移除了cmp参数，改为使用functools库中的cmp_to_key()

        # 大神代码
        return sorted(logs, key=lambda l: (0, l.split(' ', 1)[::-1]) if l[-1].isalpha() else (1,))

obj = Solution()
print(obj.reorderLogFiles(["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]))




