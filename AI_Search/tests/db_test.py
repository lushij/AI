# simple_db_view.py
import sqlite3
import os

def simple_view():
    """简单查看数据库"""
    db_path = "../local_rag_data/chroma.sqlite3"

    if not os.path.exists(db_path):
        print("数据库文件不存在")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 60)
    print("简单数据库查看")
    print("=" * 60)

    # 1. 查看集合
    print("\n📚 集合列表:")
    cursor.execute("SELECT id, name FROM collections;")
    for row in cursor.fetchall():
        print(f"  ID: {row[0]}, 名称: {row[1]}")

    # 2. 找到 ai_knowledge_base
    cursor.execute("SELECT id FROM collections WHERE name = 'ai_knowledge_base';")
    result = cursor.fetchone()

    if result:
        collection_id = result[0]
        print(f"\n✅ 找到 ai_knowledge_base, ID: {collection_id}")

        # 3. 查看对应的嵌入表
        table_name = f"embeddings_{collection_id}"
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"📄 文档数量: {count}")

        # 4. 查看文档内容
        print("\n文档内容:")
        cursor.execute(f"SELECT id, substr(document, 1, 100) as preview FROM {table_name};")
        for row in cursor.fetchall():
            print(f"  ID: {row[0]}")
            print(f"    内容: {row[1]}...")

    conn.close()

if __name__ == "__main__":
    simple_view()