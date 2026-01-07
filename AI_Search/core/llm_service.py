"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/5
    Time:下午4:46
    To change this template use File | Settings | File Templates
"""
# core/llm_service.py
import ollama
from typing import List, Optional, Dict, Generator
import json


class LLMService:
    """修复版LLM服务"""

    def __init__(self, model_name: str = "deepseek-coder:6.7b"):
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        try:
            # 将参数转换为Ollama支持的格式
            options = {}
            if 'temperature' in kwargs:
                options['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                options['num_predict'] = kwargs['max_tokens']
            if 'top_p' in kwargs:
                options['top_p'] = kwargs['top_p']

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=options if options else None
            )
            return response.response
        except Exception as e:
            return f"生成失败: {str(e)}"

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """对话"""
        try:
            # 提取Ollama支持的选项
            options = {}
            if 'temperature' in kwargs:
                options['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                options['num_predict'] = kwargs['max_tokens']
            if 'top_p' in kwargs:
                options['top_p'] = kwargs['top_p']

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options=options if options else None
            )
            return response.message.content
        except Exception as e:
            return f"对话失败: {str(e)}"

    def rag_generate(self, question: str, context: str, **kwargs) -> str:
        """RAG专用生成"""
        system_prompt = """你是一个专业的文档助手。请基于提供的上下文信息回答问题。
如果上下文没有相关信息，请说"根据提供的资料，我无法回答这个问题"。
不要编造信息，保持回答简洁准确。"""

        user_prompt = f"""上下文信息：
{context}

问题：{question}

请基于上下文信息回答问题："""

        return self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], **kwargs)

    def stream_generate(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成"""
        try:
            options = {}
            if 'temperature' in kwargs:
                options['temperature'] = kwargs['temperature']

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options=options if options else None,
                stream=True
            )

            for chunk in response:
                if hasattr(chunk, 'response'):
                    yield chunk.response
        except Exception as e:
            yield f"流式生成失败: {str(e)}"

    def test_model(self) -> bool:
        """测试模型是否可用"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt="test",
                options={'num_predict': 10}
            )
            return True
        except:
            return False