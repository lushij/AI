"""
    Created by  PyCharm
    User:lushiji
    Date:2025-12-28
    Time:14:58
    To change this template use File | Settings | File Templates
"""


def ps(data, l, r):
    while (l < r):
        tmp = data[l]
        while (l < r and data[r] >= tmp):
            r -= 1
        data[l] = data[r]
        while (l < r and data[l] <= tmp):
            l += 1
        data[r] = data[l]
        data[l] = tmp
    return l


def quick_sort(data, l, r):
    """
    快排
    :param data:
    :return:
    """
    if (l < r):
        p = ps(data, l, r)
        quick_sort(data, l, p - 1)
        quick_sort(data, p + 1, r)


if __name__ == '__main__':
    num = [2, 3, 6, 1, 3, 5]
    print("排序前")
    print(num)
    lenth = len(num) - 1
    quick_sort(num, 0, lenth)
    print('排序后')
    print(num)