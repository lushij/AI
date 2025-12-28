"""
    Created by  PyCharm
    User:lushiji
    Date:2025-12-28
    Time:14:15
    To change this template use File | Settings | File Templates
"""


# 节点
class Node:
    def __init__(self, data):
        self.data = data
        self.lc = None
        self.rc = None


# 二叉树类
class Tree:
    def __init__(self):
        self.root = None
        self.list = []  # 辅助队列层次建树

    def level_build_tree(self, node: Node):
        if not self.root:
            self.root = node
            self.list.append(node)
        else:
            self.list.append(node)
            if not self.list[0].lc:
                self.list[0].lc = node
            else:
                self.list[0].rc = node
                self.list.pop(0)

    def pre_order(self, root: Node):
        if root:  # 当前节点不为空才可以进去
            print(root.data, end=' ')
            self.pre_order(root.lc)
            self.pre_order(root.rc)

    def mid_order(self, root: Node):
        if root:  # 当前节点不为空才可以进去
            self.mid_order(root.lc)
            print(root.data, end=' ')
            self.mid_order(root.rc)

    def end_order(self, root: Node):
        if root:  # 当前节点不为空才可以进去
            self.end_order(root.lc)
            self.end_order(root.rc)
            print(root.data, end=' ')

    def level_order(self):
        help_list = []  # 辅助队列
        help_list.append(self.root)
        while help_list:
            cur = help_list.pop(0)
            print(cur.data, end=' ')
            if cur.lc:
                help_list.append(cur.lc)
            if cur.rc:
                help_list.append(cur.rc)


if __name__ == '__main__':
    tree = Tree()
    for i in range(1, 11):
        node = Node(i)
        tree.level_build_tree(node)
    print("-" * 50)
    print('前序遍历')
    tree.pre_order(tree.root)
    print()
    print("-" * 50)
    tree.level_order()
    print()
    print("-" * 50)
