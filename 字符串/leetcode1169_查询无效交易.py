#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/20 15:30
# Author: Hou hailun

"""
如果出现下述两种情况，交易 可能无效：
    交易金额超过 ¥1000
    或者，它和另一个城市中同名的另一笔交易相隔不超过 60 分钟（包含 60 分钟整）
每个交易字符串 transactions[i] 由一些用逗号分隔的值组成，这些值分别表示交易的名称，时间（以分钟计），金额以及城市。
给你一份交易清单 transactions，返回可能无效的交易列表。你可以按任何顺序返回答案。
示例 1：
输入：transactions = ["alice,20,800,mtv","alice,50,100,beijing"]
输出：["alice,20,800,mtv","alice,50,100,beijing"]
解释：第一笔交易是无效的，因为第二笔交易和它间隔不超过 60 分钟、名称相同且发生在不同的城市。同样，第二笔交易也是无效的。
示例 2：
输入：transactions = ["alice,20,800,mtv","alice,50,1200,mtv"]
输出：["alice,50,1200,mtv"]
示例 3：
输入：transactions = ["alice,20,800,mtv","bob,50,1200,mtv"]
输出：["bob,50,1200,mtv"]
"""
import collections


class Solution(object):
    def invalidTransactions(self, transactions):
        """
        :type transactions: List[str]
        :rtype: List[str]
        """
        # 先把所有的交易记录用哈希表，按照每个人的名字分好类，
        # 然后在每个人的名字下的交易里，找满足无效交易的两个条件：
        #   1.交易金额超过1000。
        #   2.有不同城市的一小时内的其他转账记录。
        # 问题: 不能保证按顺序执行
        record_by_name = collections.defaultdict(list)
        for trans in transactions:
            name, time, amount, city = trans.split(',')
            record_by_name[name].append([name, int(time), int(amount), city])

        def convert(l):  # 转换为指定格式
            return l[0] + ',' + str(l[1]) + ',' + str(l[2]) + ',' + l[3]

        res = set()
        for name, rec in record_by_name.items():
            sorted_rec = sorted(rec, key=lambda x: x[1])  # 按时间排好序
            for i in range(len(sorted_rec)):
                if sorted_rec[i][2] > 1000:  # 交易金额大于1000
                    res.add(convert(sorted_rec[i]))
                for j in range(i+1, len(sorted_rec)):
                    if abs(sorted_rec[i][1] - sorted_rec[j][1]) > 60:  # 只在两笔交易时间相距60m内的数据中找无效交易
                        break
                    if sorted_rec[i][3] != sorted_rec[j][3]:  # 不同城市
                        res.add(convert(sorted_rec[i]))
                        res.add(convert(sorted_rec[j]))
        return res


obj = Solution()
ret = obj.invalidTransactions(["alice,20,800,mtv", "alice,50,100,beijing"])
print(ret)