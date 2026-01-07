"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/5
    Time:下午3:48
    To change this template use File | Settings | File Templates
"""
# test_deepseek.py
import ollama

def test_deepseek():
    """测试DeepSeek模型"""
    print("测试DeepSeek模型...")
    models_to_test = [
        "deepseek-coder:6.7b",
    ]

    for model in models_to_test:
        try:
            print(f"\n测试模型: {model}")
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user',
                    'content': '用Python写一个快速排序算法'
                }]
            )
            print(f"结果: {response['message']['content'][:100]}...")
        except Exception as e:
            print(f"模型 {model} 不可用: {e}")


if __name__ == "__main__":
    test_deepseek()