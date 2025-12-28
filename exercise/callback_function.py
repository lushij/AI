"""
    Created by  PyCharm
    User:lushiji
    Date:2025-12-28
    Time:14:58
    To change this template use File | Settings | File Templates
"""


"""
练习sort和sorted的回调函数
sorted()（返回新列表）
使用 sort()（原地修改）
"""

def change(strs: str):
    """转换为小写"""
    return strs.lower()

def use_sorted(data, key_func):
    """使用sorted返回新列表"""
    return sorted(data, key=key_func)

def use_sort(data, key_func):
    """使用sort原地排序"""
    data.sort(key=key_func)  # 原地修改
    return data  # 返回修改后的列表

# 补充更多回调函数示例
def by_length(s: str):
    """按字符串长度排序"""
    return len(s)

def by_last_char(s: str):
    """按最后一个字符排序"""
    return s[-1].lower() if s else ''

def by_middle_char(s: str):
    """按中间字符排序（如果长度为偶数取前一个）"""
    if not s:
        return ''
    mid = len(s) // 2
    return s[mid].lower()

def by_vowel_count(s: str):
    """按元音字母数量排序"""
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

def by_custom_rule(s: str):
    """自定义规则：先按长度，再按小写字母"""
    return (len(s), s.lower())

def by_reverse(s: str):
    """按反转后的字符串排序"""
    return s[::-1].lower()

if __name__ == '__main__':
    # 原始示例
    num = ['A', 'v', 'a', 'STR']
    new_num = use_sorted(num, change)
    print("1. 原始示例 - 按小写排序:")
    print(f"   原列表: {num}")
    print(f"   排序后: {new_num}")
    print()

    # 2. 更多字符串示例
    fruits = ['Apple', 'banana', 'Cherry', 'date', 'Elderberry', 'fig']

    print("2. 按长度排序:")
    print(f"   sorted: {sorted(fruits, key=by_length)}")
    fruits_copy = fruits.copy()
    fruits_copy.sort(key=by_length)
    print(f"   sort:   {fruits_copy}")
    print()

    print("3. 按最后一个字符排序:")
    print(f"   {sorted(fruits, key=by_last_char)}")
    print()

    print("4. 按中间字符排序:")
    print(f"   {sorted(fruits, key=by_middle_char)}")
    print()

    print("5. 按元音字母数量排序:")
    for fruit in sorted(fruits, key=by_vowel_count):
        vowels = sum(1 for char in fruit if char.lower() in 'aeiou')
        print(f"   {fruit}: {vowels}个元音")
    print()

    print("6. 自定义规则（先长度，后字母）:")
    print(f"   {sorted(fruits, key=by_custom_rule)}")
    print()

    print("7. 按反转字符串排序:")
    print(f"   {sorted(fruits, key=by_reverse)}")
    print()

    # 8. 使用 lambda 表达式
    print("8. Lambda 表达式示例:")
    print(f"   a) 按第二个字符排序: {sorted(fruits, key=lambda x: x[1].lower() if len(x) > 1 else '')}")
    print(f"   b) 按字符串包含的 'a' 数量: {sorted(fruits, key=lambda x: x.lower().count('a'))}")
    print()

    # 9. 降序排序
    print("9. 降序排序:")
    print(f"   按长度降序: {sorted(fruits, key=len, reverse=True)}")
    print(f"   按字母降序: {sorted(fruits, key=str.lower, reverse=True)}")
    print()

    # 10. 复杂数据结构排序
    print("10. 复杂数据结构排序:")
    students = [
        {'name': 'Alice', 'age': 20, 'grade': 85},
        {'name': 'Bob', 'age': 22, 'grade': 90},
        {'name': 'Charlie', 'age': 21, 'grade': 78},
        {'name': 'David', 'age': 20, 'grade': 92}
    ]

    print("   按年龄排序:")
    print(f"   {sorted(students, key=lambda x: x['age'])}")
    print()

    print("   先按年龄，再按成绩排序:")
    print(f"   {sorted(students, key=lambda x: (x['age'], x['grade']))}")
    print()

    print("   按成绩降序:")
    print(f"   {sorted(students, key=lambda x: x['grade'], reverse=True)}")
    print()

    # 11. 多级排序技巧
    print("11. 多级排序技巧:")
    mixed = ['a10', 'a2', 'b1', 'b10', 'a1', 'b2']

    print(f"   默认排序: {sorted(mixed)}")  # 字符串比较

    # 按字母和数字分开排序
    def natural_sort_key(s: str):
        import re
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    print(f"   自然排序: {sorted(mixed, key=natural_sort_key)}")
    print()

    # 12. 性能对比
    print("12. sort() vs sorted() 对比:")
    import time

    large_list = list(range(10000, 0, -1))  # 创建一个大列表

    # sorted() 测试
    start = time.time()
    new_list = sorted(large_list)
    time_sorted = time.time() - start

    # sort() 测试
    large_list_copy = large_list.copy()
    start = time.time()
    large_list_copy.sort()
    time_sort = time.time() - start

    print(f"   sorted() 用时: {time_sorted:.6f}秒")
    print(f"   sort() 用时: {time_sort:.6f}秒")
    print("   注意：两者性能相近，但sorted()需要额外内存")
    print()

    # 13. 链式调用
    print("13. 链式调用示例:")
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]

    # 先排序，再反转
    result = sorted(numbers)[::-1]
    print(f"   升序后反转: {result}")

    # 或者直接使用 reverse 参数
    result2 = sorted(numbers, reverse=True)
    print(f"   直接降序: {result2}")