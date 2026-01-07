# simplest_test.py - 最简单的测试
import ollama

def simple_test():
    print("=== 简单测试 ===")

    # 方法1：直接测试生成
    print("测试生成...")
    try:
        response = ollama.generate(
            model='deepseek-coder:6.7b',
            prompt='用Python写一个hello world程序'
        )

        # 打印响应
        print("响应对象类型:", type(response))
        print("响应对象属性:", dir(response))

        if hasattr(response, 'response'):
            print("✅ 成功！响应内容:")
            print(response.response)
        else:
            print("完整响应:", response)

    except Exception as e:
        print(f"生成测试失败: {e}")

        # 尝试chat方式
        print("\n尝试chat方式...")
        try:
            response = ollama.chat(
                model='deepseek-coder:6.7b',
                messages=[{'role': 'user', 'content': '你好'}]
            )
            print("chat响应:", response)
        except Exception as e2:
            print(f"chat也失败: {e2}")

if __name__ == "__main__":
    simple_test()