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
    ret = list(map(lambda x: sum(x), matrix))  # 解释：map(lambda x: sum(x), matrix)将matrix中的每一行的元素求和，并返回一个新的列表。
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


def use_simple_list():
    """
    使用简单列表生成式生成10个整数的平方
    :return:
    """
    squares = [i ** 2 for i in range(1, 10, 2)]  # 步长为2，生成奇数的平方
    print(squares)  # [1, 9, 25, 49]


def use_complex_list():
    """
    使用复杂列表生成包括条件判断
    :return:
    """
    ret = [i for i in range(1, 11) if i % 2 == 0 and i % 3 == 0]
    print(ret)

    # 过滤字符串列表中的数字
    words = ['apple', '123', 'banana', '456', 'cherry']
    ret = [word for word in words if word.isdigit() == False]
    print(ret)
    # 将数字分类为奇偶
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ret = ['奇数' if num % 2 == 1 else '偶数' for num in numbers]
    print(ret)


if __name__ == '__main__':
    use_liseproducer()
    use_map()
    use_list()
    use_simple_list()
    use_complex_list()
