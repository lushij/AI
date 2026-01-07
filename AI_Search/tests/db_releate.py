# chroma_explain.py
import sqlite3
import os


def explain_chroma_structure():
    """详细解释ChromaDB结构"""

    print("=" * 80)
    print("CHROMADB SQLITE数据库结构完全解析")
    print("=" * 80)

    print("\n📚 第一部分：核心概念")
    print("-" * 40)
    print("""
    1. 集合 (Collection)
       - 类似数据库中的"表"或"索引"
       - 用于存储一组相关的文档
       - 示例: "ai_knowledge_base"就是一个集合

    2. 文档 (Document)
       - 存储的文本内容
       - 示例: "Python是一种高级编程语言..."

    3. 向量 (Embedding)
       - 文档的数字表示（由AI模型生成）
       - 存储在单独的索引文件中，不在SQLite里

    4. 元数据 (Metadata)
       - 文档的附加信息
       - 示例: {"category": "编程语言", "source": "Wikipedia"}

    5. 向量ID (Embedding ID)
       - 指向向量索引文件的引用
    """)

    db_path = "../local_rag_data/chroma.sqlite3"
    if not os.path.exists(db_path):
        print(f"⚠️ 数据库文件不存在: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n🔍 第二部分：实际数据库表结构")
    print("-" * 40)

    # 查看所有表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    print(f"数据库共有 {len(tables)} 个表:")
    for table in tables:
        print(f"  📋 {table}")

    # 分析每个表
    for table_name in tables:
        print(f"\n{'=' * 60}")
        print(f"表: {table_name}")
        print('=' * 60)

        # 获取表结构
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()

        print("列定义:")
        for col in columns:
            col_id, col_name, col_type, notnull, default_val, pk = col
            pk_mark = "🔑 " if pk else "  "
            print(f"  {pk_mark}{col_name:20} {col_type:15} {'NOT NULL' if notnull else ''}")

        # 查看数据示例
        print(f"\n数据示例（前2行）:")
        try:
            cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 2;")
            rows = cursor.fetchall()

            if rows:
                for i, row in enumerate(rows):
                    print(f"\n  行 {i + 1}:")
                    for j, (col, value) in enumerate(zip(columns, row)):
                        col_name = col[1]

                        # 格式化显示
                        if value is None:
                            display_value = "NULL"
                        elif isinstance(value, str):
                            if len(value) > 50:
                                display_value = f"'{value[:50]}...'"
                            else:
                                display_value = f"'{value}'"
                        elif isinstance(value, bytes):
                            display_value = f"<二进制数据 {len(value)} bytes>"
                        else:
                            display_value = str(value)

                        print(f"    {col_name:20} = {display_value}")
            else:
                print("  无数据")

        except Exception as e:
            print(f"  查询失败: {e}")

    print("\n📊 第三部分：具体表解释")
    print("-" * 40)

    # 1. collections表解释
    print("""
    1. collections表 - 集合定义表
    ----------------------------
    字段说明：
      - id: 集合的唯一标识符（UUID格式）
      - name: 集合名称（如"ai_knowledge_base"）
      - 其他字段: ChromaDB内部使用的字段

    你的数据示例：
      id: b25dee48-ddec-4aa2-9665-fb52bdb325f3
      name: ai_knowledge_base
      这表示你有一个名为"ai_knowledge_base"的集合
    """)

    # 2. embeddings_xxx表解释
    embed_tables = [t for t in tables if t.startswith('embeddings_')]
    if embed_tables:
        table_name = embed_tables[0]  # 取第一个嵌入表

        print(f"""
    2. {table_name}表 - 文档存储表
    ----------------------------
    这个表存储实际的文档内容。表名格式：embeddings_<集合ID>

    典型字段说明：
      - id: 文档的唯一标识符
      - document: 文档的文本内容
      - metadata: 文档的元数据（JSON格式）
      - embedding_id: 指向向量索引的引用
      - 其他字段: ChromaDB内部管理字段

    让我们查看你的实际文档：
    """)

        # 查看文档示例
        try:
            cursor.execute(f"""
                SELECT id, 
                       substr(document, 1, 80) as content_preview,
                       metadata,
                       embedding_id
                FROM '{table_name}' 
                LIMIT 3;
            """)

            docs = cursor.fetchall()

            for i, (doc_id, content, metadata, embedding_id) in enumerate(docs):
                print(f"\n  文档示例 {i + 1}:")
                print(f"    📄 ID: {doc_id}")
                print(f"    📝 内容: {content}...")
                print(f"    🏷️  元数据: {metadata[:50] if metadata else '无'}")
                print(f"    🔗 向量ID: {embedding_id}")

        except Exception as e:
            print(f"  查询文档失败: {e}")

    print("\n🔗 第四部分：数据流向")
    print("-" * 40)
    print("""
    你的RAG系统数据流向：

    1. 添加文档时：
       Python代码 → ChromaDB Python库 → SQLite数据库（文档文本） + 向量索引文件（向量）

    2. 搜索时：
       用户查询 → 转换为向量 → 在向量索引中搜索相似向量 → 获取文档ID → 从SQLite获取文档文本

    3. 生成答案时：
       检索到的文档 + 用户问题 → Ollama(DeepSeek) → 生成答案

    文件存储位置：
      - 文本内容: ./local_rag_data/chroma.sqlite3 (SQLite数据库)
      - 向量数据: ./local_rag_data/chroma_index/ (Faiss索引文件)
      - 配置信息: ./local_rag_data/ 下的其他文件
    """)

    # 查看存储文件
    print("\n📁 第五部分：物理文件结构")
    print("-" * 40)

    data_dir = "../local_rag_data"
    if os.path.exists(data_dir):
        print(f"目录: {os.path.abspath(data_dir)}")

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            size = os.path.getsize(item_path) if os.path.isfile(item_path) else None

            if item == "chroma.sqlite3":
                print(f"  📊 {item} - SQLite数据库 ({size / 1024 / 1024:.2f} MB)")
                print("     存储所有文档的文本内容和元数据")

            elif item == "chroma_index":
                print(f"  🔢 {item}/ - 向量索引目录")
                if os.path.isdir(item_path):
                    index_files = os.listdir(item_path)
                    for f in index_files:
                        if f.endswith('.bin'):
                            f_path = os.path.join(item_path, f)
                            f_size = os.path.getsize(f_path)
                            print(f"      └─ {f} ({f_size / 1024 / 1024:.2f} MB)")
                            print("         存储文档的向量表示，用于快速相似度搜索")

            elif item.endswith('.sqlite3') or item.endswith('.sqlite'):
                print(f"  💾 {item} - 其他SQLite文件")

            elif os.path.isfile(item_path):
                print(f"  📄 {item} ({size / 1024:.2f} KB)")

    conn.close()

    print("\n" + "=" * 80)
    print("🎯 总结：你的AI搜索引擎数据存储")
    print("=" * 80)
    print("""
    1. 知识库名称: ai_knowledge_base
    2. 存储位置: ./local_rag_data/
    3. 文本存储: SQLite数据库 (chroma.sqlite3)
    4. 向量存储: Faiss索引文件 (chroma_index/)
    5. 文档数量: 10个（你之前添加的AI相关文档）
    6. 模型配置: ./models/ 下的嵌入模型

    下次运行程序时，这些数据会自动加载，不需要重新添加！
    你可以继续添加新文档，它们会被持久化保存。
    """)


def create_simple_viewer():
    """创建一个简单的数据库查看器"""

    code = '''
# simple_db_viewer.py - 简单数据库查看器
import sqlite3
import os

class SimpleDBViewer:
    """简单的数据库查看器"""

    def __init__(self, db_path="./local_rag_data/chroma.sqlite3"):
        self.db_path = db_path

    def show_collections(self):
        """显示所有集合"""
        if not os.path.exists(self.db_path):
            print("❌ 数据库文件不存在")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print("📚 知识库集合列表")
        print("-" * 40)

        # 查看collections表的所有列
        cursor.execute("PRAGMA table_info(collections);")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"集合表列名: {columns}")

        # 查找name列（不同版本可能列名不同）
        if 'name' in columns:
            cursor.execute("SELECT id, name FROM collections;")
        else:
            # 尝试其他可能的列名
            cursor.execute("SELECT * FROM collections LIMIT 1;")
            sample = cursor.fetchone()
            if sample and len(sample) >= 2:
                print(f"集合数据示例: {sample}")
                cursor.execute("SELECT * FROM collections;")
            else:
                print("无法识别集合表结构")
                conn.close()
                return

        collections = cursor.fetchall()
        print(f"\\n找到 {len(collections)} 个集合:")

        for i, row in enumerate(collections):
            print(f"\\n[{i+1}] 集合信息:")
            for j, value in enumerate(row):
                col_name = columns[j] if j < len(columns) else f"列{j}"
                print(f"  {col_name}: {value}")

        conn.close()

    def show_documents(self, collection_name="ai_knowledge_base"):
        """显示文档内容"""
        if not os.path.exists(self.db_path):
            print("❌ 数据库文件不存在")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        print(f"\\n📄 查看集合 '{collection_name}' 的文档")
        print("-" * 40)

        # 1. 先找到集合对应的表
        # 查找所有嵌入表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'embeddings_%';")
        embed_tables = [row[0] for row in cursor.fetchall()]

        if not embed_tables:
            print("❌ 未找到文档表")
            conn.close()
            return

        # 2. 查看每个嵌入表的文档
        total_docs = 0
        for table_name in embed_tables:
            print(f"\\n检查表: {table_name}")

            # 查看表结构
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = [col[1] for col in cursor.fetchall()]

            # 查找document列
            if 'document' in columns:
                # 获取文档数量
                cursor.execute(f"SELECT COUNT(*) FROM '{table_name}';")
                count = cursor.fetchone()[0]
                total_docs += count

                print(f"  文档数量: {count}")

                if count > 0:
                    print("  文档内容:")
                    cursor.execute(f"""
                        SELECT id, 
                               substr(document, 1, 80) as preview,
                               substr(metadata, 1, 50) as meta_preview
                        FROM '{table_name}' 
                        ORDER BY id
                        LIMIT 3;
                    """)

                    for doc_id, preview, metadata in cursor.fetchall():
                        print(f"  📝 ID: {doc_id}")
                        print(f"     内容: {preview}...")
                        if metadata:
                            print(f"     元数据: {metadata}")

        print(f"\\n📊 总计: {total_docs} 个文档")
        conn.close()

# 使用示例
if __name__ == "__main__":
    viewer = SimpleDBViewer()

    print("=" * 60)
    print("AI知识库数据库查看器")
    print("=" * 60)

    viewer.show_collections()
    viewer.show_documents()

    print("\\n✅ 查看完成！")
'''

    # 保存到文件
    with open("simple_db_viewer.py", "w", encoding="utf-8") as f:
        f.write(code)

    print("\n💡 我已经创建了一个简单的查看器文件：simple_db_viewer.py")
    print("运行命令: python simple_db_viewer.py")


if __name__ == "__main__":
    # 解释结构
    explain_chroma_structure()

    # 创建查看器
    create_simple_viewer()