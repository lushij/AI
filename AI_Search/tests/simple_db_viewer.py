
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
        print(f"\n找到 {len(collections)} 个集合:")

        for i, row in enumerate(collections):
            print(f"\n[{i+1}] 集合信息:")
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

        print(f"\n📄 查看集合 '{collection_name}' 的文档")
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
            print(f"\n检查表: {table_name}")

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

        print(f"\n📊 总计: {total_docs} 个文档")
        conn.close()

# 使用示例
if __name__ == "__main__":
    viewer = SimpleDBViewer()

    print("=" * 60)
    print("AI知识库数据库查看器")
    print("=" * 60)

    viewer.show_collections()
    viewer.show_documents()

    print("\n✅ 查看完成！")
