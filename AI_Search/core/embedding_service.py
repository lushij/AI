"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/5
    Time:下午5:02
    To change this template use File | Settings | File Templates
"""
# core/embedding_service.py
import os
from sentence_transformers import SentenceTransformer
import logging


class EmbeddingService:
    """嵌入模型服务，处理下载和缓存"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh", cache_dir: str = "./models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

    def load_model(self) -> SentenceTransformer:
        """加载模型，带重试机制"""
        if self.model is not None:
            return self.model

        try:
            print(f"加载嵌入模型: {self.model_name}")

            # 尝试从缓存加载
            local_path = os.path.join(self.cache_dir, self.model_name.replace("/", "_"))
            if os.path.exists(local_path):
                print("从本地缓存加载...")
                self.model = SentenceTransformer(local_path)
            else:
                # 在线下载，增加超时时间
                print("从HuggingFace下载...")
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir,
                    device='cpu'  # 使用CPU避免GPU问题
                )

                # 保存到本地缓存
                self.model.save(local_path)
                print(f"模型已保存到: {local_path}")

            print("✅ 嵌入模型加载完成")
            return self.model

        except Exception as e:
            print(f"❌ 加载嵌入模型失败: {e}")
            print("尝试使用备用的小模型...")

            # 使用更小的备用模型
            try:
                self.model = SentenceTransformer(
                    "all-MiniLM-L6-v2",  # 英文小模型，更容易下载
                    cache_folder=self.cache_dir
                )
                print("✅ 使用备用模型: all-MiniLM-L6-v2")
                return self.model
            except:
                raise RuntimeError("无法加载任何嵌入模型")