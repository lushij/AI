"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/24
    Time:下午4:55
    To change this template use File | Settings | File Templates
"""
from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response=client.chat.completions.create(
    model="deepseek-v3.2",
    messages=[
        {"role":"system","content":"你是一个不说废话的编程助手。"},
        {"role":"assistant","content":"好的，我从不说废话，简单明了地回答你的问题"},
        {"role":"user","content":"请帮我写一个Python函数，计算两个数的和。"}
    ]
)
print(response.choices[0].message.content)

""""
ChatCompletion(id='chatcmpl-a971817e-a461-9987-9e3f-c01003399969', 
choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='```python\ndef add(a, b):\n    return a + b\n```', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))], 
created=1769245225,
model='deepseek-v3.2', 
object='chat.completion', 
service_tier=None, 
system_fingerprint=None, 
usage=CompletionUsage(completion_tokens=16, 
prompt_tokens=39, total_tokens=55, 
completion_tokens_details=None, 
prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0)))

进程已结束，退出代码为 0
"""