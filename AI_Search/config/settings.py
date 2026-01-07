"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/5
    Time:下午5:50
    To change this template use File | Settings | File Templates
"""
# config/settings.py
import os
from pathlib import Path


class Settings:
    # 路径配置
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"

    # 确保目录存在
    for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(exist_ok=True)

    # RAG配置
    RAG_CONFIG = {
        "persist_dir": str(DATA_DIR / "rag_db"),
        "embed_model": "all-MiniLM-L6-v2",
        "llm_model": "deepseek-coder:6.7b",
        "collection_name": "ai_knowledge_base",
        "top_k": 3,
        "temperature": 0.1,
        "max_tokens": 512
    }

    # 安全配置
    SECURITY_CONFIG = {
        "enable_audit": True,
        "audit_log_file": str(LOGS_DIR / "audit.log"),
        "enable_input_validation": True,
        "sensitive_keywords": ["密码", "密钥", "身份证", "银行卡"]
    }

    # API配置
    API_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": True,
        "workers": 1
    }


settings = Settings()