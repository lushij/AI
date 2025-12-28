"""
    Created by  PyCharm
    User:lushiji
    Date:2025-12-28
    Time:10:55
    To change this template use File | Settings | File Templates
"""
import os
from openai import OpenAI  # 需要先安装: pip install openai


def test():
    """
    测试
    :return:
    """
    print("This is a test function in the exercise package.")


def test_key():
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key exists: {api_key is not None}")
    print(f"API Key value (first 10 chars): {api_key[:10] if api_key else 'None'}")
    return api_key


def test_openai_api():
    """
    测试 OpenAI API 连接
    """
    print("\n=== 测试 OpenAI API 连接 ===")

    # 方法1：尝试从环境变量获取
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"使用的密钥: {api_key[:10]}...")

    try:
        # 创建客户端
        client = OpenAI(api_key=api_key)

        # 测试调用
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请用中文回答：Hello, 世界！"}
            ],
            max_tokens=100,
            temperature=0.7
        )

        print("✅ OpenAI API 连接成功！")
        print(f"回复: {response.choices[0].message.content}")
        print(f"使用 tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n可能的原因：")
        print("1. API 密钥错误或过期")
        print("2. 网络连接问题")
        print("3. 账户没有额度或需要验证")
        print("4. 密钥格式不正确")


if __name__ == '__main__':
    print("=== OpenAI API 测试程序 ===")

    # 选项1：只测试环境变量
    # test_key()
    test_openai_api()