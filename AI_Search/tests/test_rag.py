"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/5
    Time:下午4:46
    To change this template use File | Settings | File Templates
"""
# final_test.py
from core.rag_engine import RAGEngine

def main():
    print("=== RAG系统完整测试 ===")

    # 1. 初始化引擎
    print("初始化RAG引擎...")
    rag = RAGEngine(
        persist_dir="../simple_rag_data",
        llm_model="deepseek-coder:6.7b"
    )

    # 2. 列出已有集合
    collections = rag.list_collections()
    print(f"现有集合: {collections}")

    # 3. 创建新集合
    collection_name = "test_collection"
    if collection_name in collections:
        print(f"集合 '{collection_name}' 已存在，使用现有集合")
    else:
        rag.create_collection(collection_name, "测试集合")
        print(f"创建集合: {collection_name}")

    # 4. 检查集合信息
    info = rag.get_collection_info(collection_name)
    print(f"集合信息: {info}")

    # 5. 如果集合为空，添加测试文档
    if info['count'] == 0:
        print("\n添加测试文档...")
        test_docs = [
            "Python是一种高级编程语言，广泛应用于Web开发、数据科学和人工智能。",
            "机器学习使计算机能够从数据中学习模式，而无需显式编程。",
            "Ollama是一个本地大模型运行工具，支持多种开源模型。",
            "RAG技术结合检索和生成，提高AI回答的准确性。",
            "向量数据库通过向量相似度进行高效检索。"
        ]

        test_metadatas = [
            {"type": "programming", "topic": "Python"},
            {"type": "ai", "topic": "Machine Learning"},
            {"type": "tool", "topic": "Ollama"},
            {"type": "technique", "topic": "RAG"},
            {"type": "database", "topic": "Vector DB"}
        ]

        doc_ids = rag.add_documents(collection_name, test_docs, test_metadatas)
        print(f"添加了 {len(doc_ids)} 个文档")

    # 6. 重新检查集合
    info = rag.get_collection_info(collection_name)
    print(f"更新后集合信息: {info}")

    # 7. 测试搜索
    print("\n" + "=" * 40)
    print("测试搜索功能")
    print("=" * 40)

    test_queries = [
        "什么是Python？",
        "机器学习是什么？",
        "Ollama有什么作用？"
    ]

    for query in test_queries:
        print(f"\n搜索: '{query}'")
        results = rag.search(query, collection_name, top_k=2)

        print(f"找到 {results['count']} 个相关文档:")
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"  {i + 1}. {doc[:60]}...")
            if metadata:
                print(f"     元数据: {metadata}")

    # 8. 测试问答
    print("\n" + "=" * 40)
    print("测试问答功能")
    print("=" * 40)

    questions = [
        "Python能用来做什么？",
        "解释一下机器学习",
        "什么是RAG技术？"
    ]

    for question in questions:
        print(f"\n问题: {question}")

        try:
            result = rag.ask(
                question,
                collection_name,
                temperature=0.1,
                max_tokens=200
            )

            print(f"回答: {result['answer']}")
            print(f"响应时间: {result['response_time']:.2f}秒")
            print(f"参考文档数: {result['sources_count']}")

            if result['sources']:
                print("参考文档:")
                for i, source in enumerate(result['sources'][:2]):
                    print(f"  {i + 1}. {source['content_preview']}")
                    if source.get('score'):
                        print(f"     相关度: {source['score']:.2%}")

        except Exception as e:
            print(f"❌ 问答出错: {e}")
            import traceback
            traceback.print_exc()

    # 9. 测试空查询
    print("\n" + "=" * 40)
    print("测试边界情况")
    print("=" * 40)

    empty_result = rag.ask("不存在的主题", collection_name)
    print(f"空查询: {empty_result['answer']}")

    # 10. 清理测试（可选）
    print("\n" + "=" * 40)
    print("测试完成")
    print("=" * 40)

    # 如果要清理测试数据，取消注释下面的代码
    # rag.delete_collection(collection_name)
    # print("已清理测试数据")

    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    main()