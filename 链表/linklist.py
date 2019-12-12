#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/10/14 15:21
# Author: Hou hailun

# 链表是一种线性表结构，采用链式存储结构，在节点中有一个指向下一节点的指针
# 插入、删除的复杂度O(1), 查找的复杂度O(N)
# 优点：不必预先知道数据大小，灵活使用内存空间
# 单链表，循环链表，双向链表等


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


class LinkList(object):
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head

    def add(self, item):
        # 头插法
        node = ListNode(item)
        node.next = self.head
        self.head = node

    def size(self):
        # 链表节点个数
        node_cnt = 0
        cur_node = self.head
        while cur_node:
            node_cnt += 1
            cur_node = cur_node.next
        return node_cnt

    def research(self, item):
        cur_node = self.head
        is_find = False
        while cur_node:
            if cur_node.val == item:
                is_find = True
                break
            cur_node = cur_node.next
        return is_find

    def remove(self, ix):
        # 删除ix位置的节点，删除节点需要知道前一节点，待删除节点
        del_data = 0
        cur_node = self.head
        pre_node = None

        if 0 == ix:  # 删除头节点
            del_data = self.head.var
            self.head = self.head.next
            return del_data
        # 删除中间节点，则需要遍历链表
        i = 0
        while cur_node:
            if i == ix:
                pre_node.next = cur_node.next
                del_data = cur_node.var
            elif i < ix:
                pre_node = cur_node
                cur_node = cur_node.next
        # 遍历完成, 仍没有找到
        if i > ix:
            del_data = -1
        return del_data
