"""
    Created by PyCharm
    User:lushiji
    Date:2026/2/6
    Time:下午8:10
    To change this template use File | Settings | File Templates
"""
# # 原函数 1
# def shopping():
#     print("🛒 正在购物...")
#
# # 原函数 2
# def pay():
#     print("💸 正在支付...")
#
# # ----- 实际调用时，你需要写这么多重复代码 -----
# print("--- [系统] 开始记录 ---")
# shopping()
# print("--- [系统] 记录完成 ---")
#
# print("--- [系统] 开始记录 ---")
# pay()
# print("--- [系统] 记录完成 ---")

# 第一步：定义装饰器 (通用的模具)
def add_log(func):
    # wrapper 是包装纸，把原函数 func 包在里面
    def wrapper():
        print(f"--- [系统] 开始记录: {func.__name__} ---")  # 加特技：前置操作

        func()  # <--- 这里是真正的原函数在执行！

        print(f"--- [系统] 记录完成 ---\n")  # 加特技：后置操作

    return wrapper


# 第二步：使用装饰器 (贴上去就行)
@add_log
def shopping():
    print("🛒 正在购物...")


@add_log
def pay():
    print("💸 正在支付...")


"""
类似于回调函数，但不完全相同

"""





# 第三步：直接调用 (就像调用普通函数一样)
shopping()
pay()

