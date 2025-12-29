"""
    Created by PyCharm
    User:lushiji
    Date:2025/12/29
    Time:上午10:42
    To change this template use File | Settings | File Templates
"""


def use_liseproducer():
    # 二维列表
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # 使用列表生成式展平
    flattened = [item for sublist in matrix for item in sublist]
    """
    [item for sublist in matrix for item in sublist]
    #       ↑ 外层循环       ↑ 内层循环
    # 等价于：
    # result = []
    # for sublist in matrix:
    #     for item in sublist:
    #         result.append(item)
"""
    print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

def use_map():
    # 二维列表
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ret = list(map(lambda x: sum(x), matrix))#解释：map(lambda x: sum(x), matrix)将matrix中的每一行的元素求和，并返回一个新的列表。
    # 等价于：
    # result = []
    # for sublist in matrix:
    #     result.append(sum(sublist))    # 这里的sum()函数是求和函数，可以换成其他函数，如max()、min()等。
    # ret = result   # 将result列表赋值给ret变量。
    print(ret)  # [6, 15, 24]
def use_list():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ret = [i for j in matrix for i in j]
    print(ret)
if __name__ == '__main__':
    use_liseproducer()
    use_map()