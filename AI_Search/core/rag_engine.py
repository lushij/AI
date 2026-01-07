"""
    Created by PyCharm
    User: lushiji
    Date: 2026/1/5
    Time: 下午5:16
    To change this template use File | Settings | File Templates
"""
# rag_engine.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import hashlib
import os
import time
import ollama
import re
import numpy as np
from datetime import datetime


# ========== 使用本地嵌入模型 ==========
class LocalEmbeddingService:
    """使用本地下载的嵌入模型"""

    def __init__(self, model_path: str = "./models/all-MiniLM-L6-v2"):
        self.model_path = model_path
        self.model = None
        self.dim = 384  # all-MiniLM-L6-v2的维度

    def load_model(self):
        """从本地路径加载模型"""
        if self.model is not None:
            return self.model

        try:
            from sentence_transformers import SentenceTransformer

            print(f"从本地加载嵌入模型: {self.model_path}")

            if not os.path.exists(self.model_path):
                print(f"⚠️ 模型文件不存在: {self.model_path}")
                print("尝试从huggingface下载模型...")
                # 下载模型
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                model.save(self.model_path)
                self.model = model
            else:
                # 从本地加载
                self.model = SentenceTransformer(self.model_path)

            print(f"✅ 本地模型加载成功，向量维度: {self.dim}")
            return self.model

        except ImportError:
            print("❌ 未安装sentence-transformers库，请安装: pip install sentence-transformers")
            return self._create_fallback_embedder()
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return self._create_fallback_embedder()

    def _create_fallback_embedder(self):
        """创建备用嵌入器"""
        print("⚠️ 使用简单嵌入器（备用方案）")
        self.model = SimpleEmbedder(dim=self.dim)
        return self.model


class SimpleEmbedder:
    """简单的嵌入器（备用方案）"""

    def __init__(self, dim: int = 384):
        self.dim = dim
        print(f"⚠️ 使用简单嵌入器，向量维度: {dim}")

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # 基于文本生成伪随机向量（可重复）
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.randn(self.dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]


# ========== LLM服务 ==========
class LocalLLMService:
    """本地LLM服务"""

    def __init__(self, model_name: str = "deepseek-coder:6.7b"):
        self.model_name = model_name
        print(f"🤖 使用LLM模型: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        try:
            options = {}
            if 'temperature' in kwargs:
                options['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                options['num_predict'] = kwargs['max_tokens']

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
            options = {}
            if 'temperature' in kwargs:
                options['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                options['num_predict'] = kwargs['max_tokens']

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
如果上下文没有相关信息，请诚实地说不知道，不要编造信息。"""

        user_prompt = f"""上下文信息：
{context}

问题：{question}

请基于上下文信息回答问题："""

        return self.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], **kwargs)


# ========== 完整RAG引擎 ==========
class CompleteRAGEngine:
    """完整的本地RAG引擎"""

    def __init__(self,
                 persist_dir: str = "./complete_rag_data",
                 embed_model_path: str = "./models/all-MiniLM-L6-v2",
                 llm_model: str = "deepseek-coder:6.7b"):

        # 创建数据目录
        os.makedirs(persist_dir, exist_ok=True)

        # 1. 加载本地嵌入模型
        print("🔧 加载本地嵌入模型...")
        self.embed_service = LocalEmbeddingService(embed_model_path)
        self.embed_model = self.embed_service.load_model()

        # 2. 初始化向量数据库
        print("💾 初始化向量数据库...")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 3. 初始化LLM服务
        print("🤖 初始化LLM服务...")
        self.llm = LocalLLMService(llm_model)

        print("✅ RAG引擎初始化完成")

    # ========== 集合管理 ==========
    def create_collection(self, name: str, metadata: Dict = None) -> bool:
        """创建集合"""
        try:
            self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {}
            )
            print(f"✅ 创建集合: {name}")
            return True
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            return False

    def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"❌ 列出集合失败: {e}")
            return []

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            collection = self.client.get_collection(name)
            return {
                "name": name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            return {"error": str(e), "name": name}

    def delete_collection(self, name: str) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(name)
            print(f"✅ 删除集合: {name}")
            return True
        except Exception as e:
            print(f"❌ 删除集合失败: {e}")
            return False

    # ========== 文档管理 ==========
    def add_documents(self,
                      collection_name: str,
                      documents: List[str],
                      metadatas: Optional[List[Dict]] = None) -> List[str]:
        """添加文档到知识库"""
        try:
            collection = self.client.get_collection(collection_name)

            # 处理metadata
            if metadatas is None:
                metadatas = [{} for _ in range(len(documents))]
            elif len(metadatas) != len(documents):
                metadatas = [metadatas[i] if i < len(metadatas) else {}
                             for i in range(len(documents))]

            # 确保每个metadata都是字典
            metadatas = [md if isinstance(md, dict) else {} for md in metadatas]

            # 处理文档
            processed_docs = []
            processed_metadatas = []
            embeddings = []
            doc_ids = []

            for i, doc in enumerate(documents):
                if not doc or not doc.strip():
                    continue  # 跳过空文档

                processed_docs.append(doc)
                processed_metadatas.append(metadatas[i])

                # 生成文档ID
                doc_id = f"doc_{i}_{int(time.time())}_{hashlib.md5(doc.encode()).hexdigest()[:8]}"
                doc_ids.append(doc_id)

                # 生成向量
                embedding = self.embed_model.encode(doc)
                embeddings.append(embedding.tolist())

            if not processed_docs:
                print("⚠️ 没有有效的文档可添加")
                return []

            # 添加文档
            collection.add(
                documents=processed_docs,
                embeddings=embeddings,
                metadatas=processed_metadatas,
                ids=doc_ids
            )

            print(f"✅ 添加了 {len(processed_docs)} 个文档到 '{collection_name}'")
            return doc_ids

        except Exception as e:
            print(f"❌ 添加文档失败: {e}")
            return []

    def add_documents_with_chunking(self,
                                   collection_name: str,
                                   documents: List[str],
                                   metadatas: Optional[List[Dict]] = None,
                                   chunk_size: int = 300,
                                   chunk_overlap: int = 50) -> List[str]:
        """添加文档并自动分块（改进版）"""
        all_chunks = []
        all_metadatas = []

        for i, doc in enumerate(documents):
            if not doc or not doc.strip():
                continue

            # 简单分块：按句子分割
            sentences = re.split(r'[。！？；\.!?;]', doc.strip())

            current_chunk = []
            current_length = 0
            chunk_index = 0

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # 添加句号
                sentence_with_dot = sentence.strip() + '。'
                sent_length = len(sentence_with_dot)

                # 如果当前chunk太大或句子本身很长，开始新chunk
                if (current_length + sent_length > chunk_size and current_chunk) or sent_length > chunk_size:
                    # 保存当前chunk
                    chunk_text = ''.join(current_chunk)
                    all_chunks.append(chunk_text)

                    # 创建metadata
                    meta = {}
                    if metadatas and i < len(metadatas):
                        meta.update(metadatas[i] if isinstance(metadatas[i], dict) else {})
                    meta.update({
                        "original_doc_index": i,
                        "chunk_index": chunk_index,
                        "is_chunked": True,
                        "chunk_size": len(chunk_text),
                        "char_count": len(chunk_text)
                    })
                    all_metadatas.append(meta)

                    # 重置并添加当前句子（使用重叠）
                    chunk_index += 1
                    if chunk_overlap > 0 and current_chunk:
                        # 保留部分内容作为重叠
                        overlap_text = ''.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                        current_chunk = [overlap_text, sentence_with_dot]
                        current_length = len(overlap_text) + sent_length
                    else:
                        current_chunk = [sentence_with_dot]
                        current_length = sent_length
                else:
                    current_chunk.append(sentence_with_dot)
                    current_length += sent_length

            # 处理最后一个chunk
            if current_chunk:
                chunk_text = ''.join(current_chunk)
                all_chunks.append(chunk_text)
                meta = {}
                if metadatas and i < len(metadatas):
                    meta.update(metadatas[i] if isinstance(metadatas[i], dict) else {})
                meta.update({
                    "original_doc_index": i,
                    "chunk_index": chunk_index,
                    "is_chunked": True,
                    "chunk_size": len(chunk_text),
                    "char_count": len(chunk_text)
                })
                all_metadatas.append(meta)

        print(f"📊 文档分块: {len(documents)} 篇文档 → {len(all_chunks)} 个chunks")

        # 使用原有的add_documents方法
        return self.add_documents(collection_name, all_chunks, all_metadatas)

    def delete_documents(self, collection_name: str, doc_ids: List[str]) -> bool:
        """删除文档"""
        try:
            collection = self.client.get_collection(collection_name)
            collection.delete(ids=doc_ids)
            print(f"✅ 删除了 {len(doc_ids)} 个文档")
            return True
        except Exception as e:
            print(f"❌ 删除文档失败: {e}")
            return False

    # ========== 查询增强 ==========
    def enhance_query(self, query: str) -> List[str]:
        """查询增强：生成相关查询"""
        enhanced_queries = [query]

        # 简单同义词扩展
        synonym_map = {
            "python": ["Python语言", "Python编程", "蟒蛇语言"],
            "机器学习": ["ML", "machine learning", "机械学习", "统计学习"],
            "深度学习": ["DL", "deep learning", "神经网络", "深度神经网络"],
            "自然语言处理": ["NLP", "自然语言理解", "语言处理"],
            "计算机视觉": ["CV", "图像识别", "视觉处理"],
            "强化学习": ["RL", "强化算法", "奖励学习"],
            "rag": ["检索增强生成", "检索式生成", "RAG技术", "检索生成"],
            "ollama": ["ollama工具", "本地大模型", "离线大模型"],
            "向量数据库": ["向量存储", "向量检索", "embedding数据库", "相似性搜索"]
        }

        query_lower = query.lower()
        for term, synonyms in synonym_map.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # 每个术语最多加2个同义词
                    enhanced_query = query.replace(term, syn) if term in query else f"{query} {syn}"
                    enhanced_queries.append(enhanced_query)

        # 添加通用扩展
        question_words = ["什么是", "解释", "介绍", "如何", "为什么"]
        if not any(qw in query for qw in question_words):
            for qw in question_words[:2]:
                enhanced_queries.append(f"{qw}{query}")

        # 去重
        unique_queries = []
        seen = set()
        for q in enhanced_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        print(f"🔍 查询扩展: '{query}' → {len(unique_queries)} 个相关查询")
        return unique_queries

    def calculate_query_similarity(self, query1: str, query2: str) -> float:
        """计算两个查询的相似度"""
        if query1 == query2:
            return 1.0

        # 基于词重叠的相似度
        words1 = set(re.findall(r'[\w\u4e00-\u9fff]+', query1.lower()))
        words2 = set(re.findall(r'[\w\u4e00-\u9fff]+', query2.lower()))

        if not words1 or not words2:
            return 0.5  # 默认值

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0

    # ========== 搜索功能 ==========
    def search(self,
               query: str,
               collection_name: str,
               top_k: int = 5) -> Dict[str, Any]:
        """搜索相关文档"""
        try:
            collection = self.client.get_collection(collection_name)

            if collection.count() == 0:
                return {
                    "query": query,
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                    "ids": [],
                    "count": 0,
                    "success": True
                }

            # 生成查询向量
            query_embedding = self.embed_model.encode(query)

            # 执行搜索
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, collection.count())
            )

            return {
                "query": query,
                "documents": results['documents'][0] if results['documents'] else [],
                "metadatas": results['metadatas'][0] if results['metadatas'] else [],
                "distances": results['distances'][0] if results['distances'] else [],
                "ids": results['ids'][0] if results['ids'] else [],
                "count": len(results['documents'][0]) if results['documents'] else 0,
                "success": True
            }

        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return {
                "query": query,
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
                "count": 0,
                "success": False,
                "error": str(e)
            }

    def search_enhanced(self,
                       query: str,
                       collection_name: str,
                       top_k: int = 5) -> Dict[str, Any]:
        """增强版搜索：多查询搜索 + 结果合并"""

        # 1. 查询扩展
        enhanced_queries = self.enhance_query(query)

        # 2. 执行多查询搜索
        all_results = {
            'documents': [],
            'metadatas': [],
            'distances': [],
            'ids': []
        }

        seen_docs = set()

        for q in enhanced_queries[:3]:  # 最多用3个查询
            try:
                results = self.search(q, collection_name, top_k=top_k * 2)

                if results.get('success') and results.get('documents'):
                    for i, doc in enumerate(results['documents']):
                        doc_id = results['ids'][i] if i < len(results.get('ids', [])) else doc[:50]

                        # 去重
                        if doc_id in seen_docs:
                            continue
                        seen_docs.add(doc_id)

                        # 计算增强分数（考虑查询相似性）
                        base_score = 1.0
                        if i < len(results.get('distances', [])):
                            base_score = max(0, 1.0 - results['distances'][i])

                        # 查询相似性权重
                        query_sim = self.calculate_query_similarity(query, q)
                        final_score = base_score * (0.7 + 0.3 * query_sim)  # 加权

                        all_results['documents'].append(doc)
                        all_results['metadatas'].append(
                            results['metadatas'][i] if i < len(results.get('metadatas', [])) else {}
                        )
                        all_results['distances'].append(1.0 - final_score)  # 转回距离
                        all_results['ids'].append(doc_id)
            except Exception as e:
                print(f"⚠️ 查询'{q}'搜索失败: {e}")
                continue

        # 3. 按分数排序
        if all_results['distances']:
            sorted_indices = sorted(range(len(all_results['distances'])),
                                    key=lambda i: all_results['distances'][i])

            sorted_results = {
                'query': query,
                'documents': [all_results['documents'][i] for i in sorted_indices[:top_k]],
                'metadatas': [all_results['metadatas'][i] for i in sorted_indices[:top_k]],
                'distances': [all_results['distances'][i] for i in sorted_indices[:top_k]],
                'ids': [all_results['ids'][i] for i in sorted_indices[:top_k]],
                'count': min(len(all_results['documents']), top_k),
                'success': True,
                'enhanced': True
            }
        else:
            # 回退到普通搜索
            sorted_results = self.search(query, collection_name, top_k)
            sorted_results['enhanced'] = False

        return sorted_results

    # ========== 问答功能 ==========
    def ask(self,
            question: str,
            collection_name: str,
            top_k: int = 3,
            temperature: float = 0.1,
            max_tokens: int = 512) -> Dict[str, Any]:
        """智能问答"""
        start_time = time.time()

        print(f"🔍 处理问题: '{question}'")

        # 1. 检索相关文档
        search_results = self.search(question, collection_name, top_k)

        if not search_results["success"] or search_results["count"] == 0:
            elapsed = time.time() - start_time
            return {
                "question": question,
                "answer": "没有找到相关信息。",
                "sources": [],
                "sources_count": 0,
                "search_success": False,
                "response_time": elapsed,
                "model_used": self.llm.model_name
            }

        print(f"📚 找到 {search_results['count']} 个相关文档")

        # 2. 构建上下文
        context_parts = []
        for i, doc in enumerate(search_results["documents"]):
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            context_parts.append(f"[文档 {i + 1}]\n{doc}")
            print(f"  文档 {i + 1}: {doc_preview}")

        context = "\n\n".join(context_parts)

        # 3. 使用LLM生成答案
        print("🤖 使用DeepSeek生成答案...")
        try:
            answer = self.llm.rag_generate(
                question,
                context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            print("✅ 答案生成成功")
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            print(f"❌ 生成答案失败: {e}")

        # 4. 构建响应
        sources = []
        for i, (doc, metadata) in enumerate(zip(
                search_results["documents"],
                search_results["metadatas"]
        )):
            doc_id = search_results["ids"][i] if i < len(search_results["ids"]) else f"doc_{i}"

            # 计算相关度分数
            score = 1.0
            if i < len(search_results["distances"]):
                distance = search_results["distances"][i]
                score = max(0, 1.0 - distance)

            sources.append({
                "id": doc_id,
                "content_preview": doc[:100] + ("..." if len(doc) > 100 else ""),
                "metadata": metadata if metadata else {},
                "score": round(score, 3)
            })

        elapsed = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "context_preview": context[:200] + ("..." if len(context) > 200 else ""),
            "sources": sources,
            "sources_count": len(sources),
            "search_success": True,
            "response_time": round(elapsed, 2),
            "model_used": self.llm.model_name,
            "embedding_model": "all-MiniLM-L6-v2"
        }

    def ask_enhanced(self,
                    question: str,
                    collection_name: str,
                    top_k: int = 3,
                    temperature: float = 0.1,
                    max_tokens: int = 512,
                    use_enhanced_search: bool = True) -> Dict[str, Any]:
        """增强版问答"""
        start_time = time.time()

        print(f"🔍 处理问题: '{question}'")

        # 1. 检索相关文档（使用增强或普通搜索）
        if use_enhanced_search:
            search_results = self.search_enhanced(question, collection_name, top_k)
            search_type = "增强搜索"
        else:
            search_results = self.search(question, collection_name, top_k)
            search_type = "普通搜索"

        if not search_results["success"] or search_results["count"] == 0:
            elapsed = time.time() - start_time
            return {
                "question": question,
                "answer": "没有找到相关信息。",
                "sources": [],
                "sources_count": 0,
                "search_success": False,
                "response_time": elapsed,
                "model_used": self.llm.model_name,
                "search_type": search_type
            }

        print(f"📚 找到 {search_results['count']} 个相关文档 ({search_type})")

        # 2. 构建上下文（改进版）
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(
                search_results["documents"],
                search_results["metadatas"]
        )):
            # 获取文档来源信息
            source_info = ""
            if metadata.get('is_chunked'):
                source_info = f" [分块文档 {metadata.get('original_doc_index', '?') + 1}.{metadata.get('chunk_index', 0) + 1}]"
            elif metadata.get('topic'):
                source_info = f" [主题: {metadata.get('topic')}]"

            # 添加到上下文
            context_parts.append(f"[文档 {i + 1}{source_info}]\n{doc}")

            # 显示预览
            doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
            score = 1.0
            if i < len(search_results["distances"]):
                score = max(0, 1.0 - search_results["distances"][i])
            print(f"  文档 {i + 1}: [{score:.1%}] {doc_preview}")

        context = "\n\n".join(context_parts)

        # 3. 改进的提示词
        system_prompt = """你是一个专业的AI助手，请严格按照以下规则回答：
1. 只使用提供的上下文信息来回答问题
2. 每个重要观点必须注明来源文档编号，如[文档1]
3. 如果上下文信息不充分，请说明"根据已有资料，..."
4. 不要编造任何上下文之外的信息
5. 如果上下文完全没有相关信息，请诚实地说"没有找到相关信息"

请基于以下上下文信息回答问题："""

        user_prompt = f"""上下文信息：
{context}

问题：{question}

请严格基于上下文信息，并注明引用来源："""

        # 4. 使用LLM生成答案
        print("🤖 使用DeepSeek生成答案...")
        try:
            answer = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], temperature=temperature, max_tokens=max_tokens)
            print("✅ 答案生成成功")
        except Exception as e:
            answer = f"生成答案时出错: {str(e)}"
            print(f"❌ 生成答案失败: {e}")

        # 5. 构建响应
        sources = []
        for i, (doc, metadata) in enumerate(zip(
                search_results["documents"],
                search_results["metadatas"]
        )):
            doc_id = search_results["ids"][i] if i < len(search_results["ids"]) else f"doc_{i}"

            # 计算相关度分数
            score = 1.0
            if i < len(search_results["distances"]):
                distance = search_results["distances"][i]
                score = max(0, 1.0 - distance)

            sources.append({
                "id": doc_id,
                "content_preview": doc[:100] + ("..." if len(doc) > 100 else ""),
                "metadata": metadata if metadata else {},
                "score": round(score, 3),
                "is_chunked": metadata.get('is_chunked', False)
            })

        elapsed = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "context_preview": context[:200] + ("..." if len(context) > 200 else ""),
            "sources": sources,
            "sources_count": len(sources),
            "search_success": True,
            "response_time": round(elapsed, 2),
            "model_used": self.llm.model_name,
            "embedding_model": "all-MiniLM-L6-v2",
            "search_type": search_type,
            "enhanced": use_enhanced_search
        }


# ========== 使用示例 ==========
def main():
    """主测试函数"""
    print("=" * 60)
    print("本地RAG系统测试")
    print("=" * 60)

    # 1. 初始化引擎
    print("\n1. 初始化RAG引擎...")
    rag = CompleteRAGEngine(
        persist_dir="./complete_rag_data",
        embed_model_path="./models/all-MiniLM-L6-v2",
        llm_model="deepseek-coder:6.7b"
    )

    # 2. 创建或获取知识库
    collection_name = "ai_knowledge_base"
    collections = rag.list_collections()

    if collection_name in collections:
        print(f"✅ 使用现有知识库: {collection_name}")
    else:
        print(f"📁 创建新知识库: {collection_name}")
        rag.create_collection(collection_name, {
            "description": "AI相关知识库",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "技术文档",
            "language": "中文"
        })

    # 3. 检查知识库状态
    info = rag.get_collection_info(collection_name)
    print(f"📊 知识库信息: {info}")

    # 4. 如果知识库为空，添加文档
    if info["count"] == 0:
        print("\n📝 知识库为空，添加示例文档...")

        ai_documents = [
            "Python是一种高级编程语言，以简洁易读著称，广泛应用于Web开发、数据分析和人工智能。",
            "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需进行明确的编程。",
            "深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的学习过程。",
            "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
            "计算机视觉是人工智能的一个分支，使计算机能够从图像和视频中理解和提取信息。",
            "强化学习是一种机器学习方法，智能体通过与环境交互来学习最优行为策略。",
            "大型语言模型（如GPT系列）是基于Transformer架构的深度学习模型，能够生成类人文本。",
            "Ollama是一个开源项目，允许用户在本地运行大型语言模型，支持多种模型格式。",
            "RAG（检索增强生成）技术结合了信息检索和文本生成，提高了AI回答的准确性和相关性。",
            "向量数据库专门用于存储和检索高维向量数据，常用于相似性搜索和推荐系统。"
        ]

        doc_metadatas = [
            {"category": "编程语言", "topic": "Python", "level": "初级", "source": "示例"},
            {"category": "人工智能", "topic": "机器学习", "level": "中级", "source": "示例"},
            {"category": "人工智能", "topic": "深度学习", "level": "高级", "source": "示例"},
            {"category": "人工智能", "topic": "NLP", "level": "中级", "source": "示例"},
            {"category": "人工智能", "topic": "计算机视觉", "level": "中级", "source": "示例"},
            {"category": "人工智能", "topic": "强化学习", "level": "高级", "source": "示例"},
            {"category": "AI模型", "topic": "大语言模型", "level": "高级", "source": "示例"},
            {"category": "工具", "topic": "Ollama", "level": "中级", "source": "示例"},
            {"category": "技术", "topic": "RAG", "level": "高级", "source": "示例"},
            {"category": "数据库", "topic": "向量数据库", "level": "中级", "source": "示例"}
        ]

        # 使用分块功能添加文档
        doc_ids = rag.add_documents_with_chunking(
            collection_name,
            ai_documents,
            doc_metadatas,
            chunk_size=200,
            chunk_overlap=30
        )
        print(f"✅ 添加了 {len(doc_ids)} 个文档chunks")

        # 重新检查
        info = rag.get_collection_info(collection_name)
        print(f"📊 更新后文档数: {info['count']}")

    # 5. 测试问答系统
    print("\n" + "=" * 60)
    print("🤔 测试问答系统")
    print("=" * 60)

    test_questions = [
        "什么是Python？",
        "机器学习是什么？",
        "解释一下深度学习",
        "什么是RAG技术？",
        "Ollama有什么用途？",
        "向量数据库有什么特点？"
    ]

    for i, question in enumerate(test_questions):
        print(f"\n{i + 1}. 问题: {question}")
        print("-" * 40)

        result = rag.ask_enhanced(
            question=question,
            collection_name=collection_name,
            top_k=3,
            temperature=0.1,
            max_tokens=256,
            use_enhanced_search=True
        )

        # 显示结果
        print(f"🤖 回答: {result['answer'][:100]}..." if len(result['answer']) > 100 else f"🤖 回答: {result['answer']}")
        print(f"⏱️  响应时间: {result['response_time']}秒")
        print(f"📚 参考文档: {result['sources_count']}个")
        print(f"🔧 使用模型: {result['model_used']}")
        print(f"🔍 搜索类型: {result['search_type']}")

        # 显示相关文档
        if result['sources']:
            print("📄 相关文档:")
            for j, source in enumerate(result['sources']):
                print(f"  {j + 1}. [{source['score']:.1%}] {source['content_preview']}")

        print()

    # 6. 测试复杂查询
    print("\n" + "=" * 60)
    print("🧠 测试复杂查询")
    print("=" * 60)

    complex_queries = [
        "比较机器学习和深度学习的区别",
        "Python在人工智能中有什么应用？",
        "解释Ollama和RAG的关系"
    ]

    for query in complex_queries:
        print(f"\n查询: {query}")
        result = rag.ask_enhanced(query, collection_name, top_k=5, max_tokens=300)

        # 简洁显示
        answer_preview = result['answer']
        if len(answer_preview) > 150:
            answer_preview = answer_preview[:150] + "..."

        print(f"回答: {answer_preview}")
        print(f"参考文档数: {result['sources_count']}")
        print(f"搜索类型: {result['search_type']}")

    # 7. 系统总结
    print("\n" + "=" * 60)
    print("📊 系统总结")
    print("=" * 60)

    info = rag.get_collection_info(collection_name)
    collections_list = rag.list_collections()

    print(f"📁 知识库列表: {collections_list}")
    print(f"📂 当前知识库: {collection_name}")
    print(f"📄 文档数量: {info['count']}")
    print(f"🧠 嵌入模型: all-MiniLM-L6-v2")
    print(f"🤖 LLM模型: deepseek-coder:6.7b")
    print(f"💾 数据存储: ./complete_rag_data")

    # 8. 清理选项（测试用）
    print("\n" + "=" * 60)
    print("🧹 清理选项")
    print("=" * 60)

    # 可以选择是否清理测试数据
    cleanup = input("是否清理测试数据？(y/N): ").lower()
    if cleanup == 'y':
        rag.delete_collection(collection_name)
        print("✅ 测试数据已清理")
    else:
        print("✅ 测试数据保留在本地")

    print("\n" + "=" * 60)
    print("🎉 测试完成！本地RAG系统工作正常")
    print("=" * 60)


if __name__ == "__main__":
    # 检查必要的依赖
    try:
        import chromadb
        import ollama
        import numpy as np
        main()
    except ImportError as e:
        print(f"❌ 缺少必要依赖: {e}")
        print("请安装以下依赖：")
        print("pip install chromadb ollama sentence-transformers numpy")
        print("\n安装ollama: 请访问 https://ollama.ai 下载并运行")
        print("安装后运行: ollama pull deepseek-coder:6.7b")