# web/app.py
import streamlit as st
import sys
import os
from pathlib import Path
import time

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
web_dir = current_file.parent  # web目录
project_root = web_dir.parent  # 项目根目录 (AI_Search目录)

print(f"📁 项目根目录: {project_root}")
print(f"📁 Web目录: {web_dir}")

# 将项目根目录和core目录添加到Python路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

# 页面配置
st.set_page_config(
    page_title="AI智能知识库系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .doc-card {
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8fafc;
        border-radius: 0.5rem;
    }
    .ai-answer {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
        line-height: 1.6;
        font-size: 16px;
    }
    .code-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Monaco', 'Consolas', monospace;
        overflow-x: auto;
        margin: 1rem 0;
    }
    .stSpinner > div {
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">📚 AI智能知识库系统</h1>', unsafe_allow_html=True)
st.markdown("---")


# 检查依赖
def check_dependencies():
    """检查必要的依赖是否安装"""
    dependencies = {
        "streamlit": "前端框架",
        "chromadb": "向量数据库",
        "sentence_transformers": "嵌入模型",
        "ollama": "本地LLM",
        "numpy": "数值计算"
    }

    missing = []
    for package, desc in dependencies.items():
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append((package, desc))

    return missing


# 初始化RAG引擎（使用缓存）
@st.cache_resource(show_spinner="初始化RAG引擎...")
def get_rag_engine():
    """初始化RAG引擎"""
    try:
        print(f"🔧 正在初始化RAG引擎...")
        print(f"📁 当前工作目录: {os.getcwd()}")
        print(f"📁 Python路径:")
        for path in sys.path:
            print(f"  - {path}")

        # 尝试导入RAG引擎
        try:
            # 尝试从core目录导入
            from core.rag_engine import CompleteRAGEngine
            print("✅ 从core.rag_engine导入成功")
        except ImportError as e1:
            print(f"❌ 从core.rag_engine导入失败: {e1}")
            try:
                # 尝试直接导入rag_engine
                import rag_engine
                from rag_engine import CompleteRAGEngine
                print("✅ 从rag_engine导入成功")
            except ImportError as e2:
                print(f"❌ 从rag_engine导入失败: {e2}")
                # 尝试在core目录中寻找文件
                rag_engine_path = project_root / "core" / "rag_engine.py"
                if rag_engine_path.exists():
                    print(f"📄 找到rag_engine.py文件: {rag_engine_path}")
                    # 手动加载模块
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("rag_engine", rag_engine_path)
                    rag_engine_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(rag_engine_module)
                    CompleteRAGEngine = rag_engine_module.CompleteRAGEngine
                    print("✅ 通过文件路径导入成功")
                else:
                    # 列出core目录内容
                    core_dir = project_root / "core"
                    if core_dir.exists():
                        print(f"📂 core目录内容:")
                        for item in os.listdir(core_dir):
                            print(f"  - {item}")
                    raise ImportError(f"找不到rag_engine.py文件。请确保文件存在于: {rag_engine_path}")

        # 确保模型目录存在
        models_dir = project_root / "models" / "all-MiniLM-L6-v2"
        os.makedirs(models_dir, exist_ok=True)

        # 设置持久化目录
        persist_dir = str(project_root / "complete_rag_data")
        os.makedirs(persist_dir, exist_ok=True)

        print(f"🔧 RAG引擎参数配置:")
        print(f"  - persist_dir: {persist_dir}")
        print(f"  - embed_model_path: {models_dir}")
        print(f"  - llm_model: deepseek-coder:6.7b")

        # 初始化引擎
        rag = CompleteRAGEngine(
            persist_dir=persist_dir,
            embed_model_path=str(models_dir),
            llm_model="deepseek-coder:6.7b"
        )

        print("✅ RAG引擎初始化成功")
        return rag

    except Exception as e:
        print(f"❌ RAG引擎初始化失败: {str(e)}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        return None


def ensure_collection_exists(rag_engine, collection_name="ai_knowledge_base"):
    """确保集合存在"""
    try:
        collections = rag_engine.list_collections()
        if collection_name not in collections:
            from datetime import datetime
            success = rag_engine.create_collection(collection_name, {
                "description": "AI相关知识库",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "技术文档",
                "language": "中文"
            })
            if success:
                return True, f"✅ 创建新集合: {collection_name}"
            else:
                return False, f"❌ 创建集合失败"
        return True, f"✅ 集合已存在: {collection_name}"
    except Exception as e:
        return False, f"❌ 确保集合存在失败: {str(e)}"


def add_example_documents(rag_engine, collection_name="ai_knowledge_base"):
    """添加示例文档"""
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

    try:
        # 首先检查集合是否存在
        success, message = ensure_collection_exists(rag_engine, collection_name)
        if not success:
            print(f"创建集合失败: {message}")
            return 0

        doc_ids = rag_engine.add_documents_with_chunking(
            collection_name,
            ai_documents,
            doc_metadatas,
            chunk_size=200,
            chunk_overlap=30
        )
        return len(doc_ids) if doc_ids else 0
    except Exception as e:
        print(f"添加示例文档失败: {e}")
        # 尝试普通添加
        try:
            doc_ids = rag_engine.add_documents(
                collection_name,
                ai_documents,
                doc_metadatas
            )
            return len(doc_ids) if doc_ids else 0
        except Exception as e2:
            print(f"普通添加也失败: {e2}")
            return 0


def display_file_tree():
    """显示项目文件树"""
    st.subheader("📁 项目文件结构")

    tree = []
    for root, dirs, files in os.walk(project_root, topdown=True):
        # 排除一些目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', '.git']]

        level = root.replace(str(project_root), '').count(os.sep)
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")

        subindent = ' ' * 4 * (level + 1)
        for file in files[:10]:  # 只显示前10个文件
            if file.endswith('.py') or file.endswith('.md') or file.endswith('.txt'):
                tree.append(f"{subindent}{file}")

    with st.expander("查看详细文件结构", expanded=False):
        st.code("\n".join(tree), language="text")


def main():
    """主应用"""
    # 显示依赖检查
    missing_deps = check_dependencies()
    if missing_deps:
        st.error("❌ 缺少必要的依赖包！")
        with st.expander("查看缺少的依赖", expanded=True):
            st.write("请安装以下依赖包:")
            for package, desc in missing_deps:
                st.code(f"pip install {package}", language="bash")
                st.write(f"  - {package}: {desc}")

        if st.button("尝试自动安装依赖", key="auto_install_deps"):
            for package, _ in missing_deps:
                st.info(f"正在安装 {package}...")
                # 这里可以添加自动安装逻辑
            st.warning("请手动运行上述命令安装依赖")
        return

    # 侧边栏 - 系统管理
    with st.sidebar:
        st.header("⚙️ 系统控制")

        # 显示文件结构
        if st.button("📁 查看文件结构", use_container_width=True, key="show_file_tree"):
            display_file_tree()

        # 刷新按钮
        if st.button("🔄 刷新系统", use_container_width=True, help="清除缓存并重新加载", key="refresh_system"):
            st.cache_resource.clear()
            st.rerun()

        st.divider()

        # 系统状态
        st.header("📊 系统状态")

        # 初始化引擎
        rag_engine = get_rag_engine()

        if rag_engine:
            st.success("✅ RAG引擎已初始化")

            # 确保默认集合存在
            success, message = ensure_collection_exists(rag_engine, "ai_knowledge_base")
            if not success:
                st.error(message)
            else:
                st.info(message)

            # 显示集合信息
            try:
                collections = rag_engine.list_collections()
                if collections:
                    selected_collection = st.selectbox(
                        "📚 选择知识库",
                        collections,
                        index=0 if "ai_knowledge_base" in collections else 0,
                        help="选择要查询的知识库",
                        key="collection_selector"
                    )

                    # 集合信息
                    try:
                        info = rag_engine.get_collection_info(selected_collection)
                        if "error" not in info:
                            with st.expander(f"📋 知识库详情: {selected_collection}", expanded=False):
                                st.metric("文档数量", info.get('count', 0))
                                if info.get('metadata'):
                                    st.write("元数据:")
                                    st.json(info['metadata'])
                        else:
                            st.warning(f"获取信息失败: {info.get('error')}")
                    except Exception as e:
                        st.warning(f"获取集合信息异常: {str(e)}")
                else:
                    st.warning("📭 没有可用的知识库")
            except Exception as e:
                st.warning(f"获取知识库列表失败: {str(e)}")

        else:
            st.error("❌ RAG引擎初始化失败")

            # 显示详细错误信息
            with st.expander("🛠️ 故障排除指南", expanded=True):
                st.write("### 常见问题解决方案:")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**1. 检查依赖安装**")
                    st.code("""
pip install streamlit chromadb sentence-transformers ollama numpy
# 下载ollama并运行
# 访问 https://ollama.ai 下载并运行
ollama pull deepseek-coder:6.7b
                    """, language="bash")

                with col2:
                    st.write("**2. 检查文件结构**")
                    st.code(f"""
# 项目应有以下结构
{project_root}/
├── core/
│   └── rag_engine.py    # RAG引擎主文件
├── web/
│   └── app.py           # 当前文件
├── models/              # 模型目录
└── complete_rag_data/   # 数据目录
                    """, language="text")

        st.divider()

        # 知识库管理
        st.header("📁 知识库管理")

        if rag_engine:
            # 创建新集合
            with st.expander("新建知识库", expanded=False):
                new_collection_name = st.text_input("知识库名称", placeholder="my_knowledge_base", key="new_col_name")
                new_collection_desc = st.text_input("描述", placeholder="描述这个知识库的用途", key="new_col_desc")

                if st.button("📦 创建知识库", use_container_width=True,
                             disabled=not new_collection_name.strip(), key="create_collection_btn"):
                    try:
                        from datetime import datetime
                        metadata = {
                            "description": new_collection_desc or "用户创建的知识库",
                            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "通用",
                            "language": "中文"
                        }

                        success = rag_engine.create_collection(new_collection_name.strip(), metadata)
                        if success:
                            st.success(f"✅ 创建成功: {new_collection_name}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ 创建失败")
                    except Exception as e:
                        st.error(f"创建失败: {str(e)}")

            # 添加示例数据
            if st.button("📥 添加示例数据", use_container_width=True,
                         help="添加AI相关的示例文档", key="add_example_data"):
                if rag_engine:
                    with st.spinner("正在添加示例数据..."):
                        doc_count = add_example_documents(rag_engine)
                        if doc_count > 0:
                            st.success(f"✅ 添加了 {doc_count} 个文档chunks")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ 添加示例数据失败")

        st.divider()

        # 文档管理
        st.header("📄 文档管理")

        if rag_engine:
            with st.form("add_document_form", clear_on_submit=True):
                st.subheader("📝 添加文档")
                doc_text = st.text_area("文档内容:", height=150,
                                        placeholder="输入要添加到知识库的文本内容...", key="doc_text_area")

                col1, col2 = st.columns(2)
                with col1:
                    doc_category = st.text_input("分类:", placeholder="AI/编程/其他", key="doc_category_input")
                with col2:
                    doc_topic = st.text_input("主题:", placeholder="Python/机器学习等", key="doc_topic_input")

                submitted = st.form_submit_button("📤 添加到知识库", use_container_width=True, key="add_document_btn")

                if submitted and doc_text.strip():
                    with st.spinner("添加文档中..."):
                        success, message = ensure_collection_exists(rag_engine)
                        if success:
                            metadata = {
                                "category": doc_category or "未分类",
                                "topic": doc_topic or "未指定",
                                "source": "手动输入",
                                "type": "text",
                                "added_at": time.strftime("%Y-%m-%d %H:%M:%S")
                            }

                            try:
                                # 使用 add_documents
                                doc_ids = rag_engine.add_documents(
                                    collection_name="ai_knowledge_base",
                                    documents=[doc_text.strip()],
                                    metadatas=[metadata]
                                )

                                if doc_ids and len(doc_ids) > 0:
                                    st.success(f"✅ 添加成功！文档ID: {doc_ids[0]}")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ 添加失败，返回空ID列表")
                            except Exception as e:
                                st.error(f"❌ 添加文档失败: {str(e)}")
                        else:
                            st.error(message)

        st.divider()

        # 系统信息
        st.header("ℹ️ 系统信息")

        with st.expander("配置详情", expanded=False):
            st.info(f"""
            **当前配置：**
            - 前端框架: Streamlit
            - 向量数据库: ChromaDB
            - 嵌入模型: all-MiniLM-L6-v2 (本地)
            - LLM模型: deepseek-coder:6.7b (通过Ollama)

            **存储路径：**
            - 项目根目录: {project_root}
            - 数据目录: {project_root}/complete_rag_data/
            - 模型目录: {project_root}/models/
            """)

            # 显示Python路径
            st.write("**Python路径:**")
            st.code("\n".join(sys.path[:10]), language="python")  # 只显示前10个

    # 主界面
    if not rag_engine:
        st.error("⚠️ RAG引擎初始化失败，无法继续")

        # 显示项目结构帮助
        display_file_tree()

        # 显示可能的解决方案
        with st.expander("🛠️ 详细故障排除", expanded=True):
            st.write("### 常见问题:")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**1. 文件位置不正确**")
                st.write("确保 `rag_engine.py` 在以下位置之一:")
                st.write(f"- `{project_root}/core/rag_engine.py`")

                # 检查文件是否存在
                rag_engine_path = project_root / "core" / "rag_engine.py"

                if rag_engine_path.exists():
                    st.success(f"✅ 找到: {rag_engine_path}")
                else:
                    st.warning(f"❌ 未找到: {rag_engine_path}")

                    # 检查core目录是否存在
                    core_dir = project_root / "core"
                    if core_dir.exists():
                        st.write(f"core目录内容:")
                        for item in os.listdir(core_dir):
                            st.write(f"  - {item}")

            with col2:
                st.write("**2. Ollama未运行**")
                st.write("确保Ollama服务正在运行:")
                st.code("""
# 检查Ollama状态
ollama list

# 如果未运行，启动Ollama
ollama serve
                """, language="bash")

                st.write("**3. 模型未下载**")
                st.code("""
# 下载需要的模型
ollama pull deepseek-coder:6.7b
                """, language="bash")

        return

    # 主内容区
    tab1, tab2, tab3 = st.tabs(["🤖 智能问答", "🔍 文档搜索", "📊 系统监控"])

    with tab1:
        st.header("🤖 智能问答系统")

        # 查询设置
        with st.expander("⚙️ 查询设置", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                top_k = st.slider("检索文档数", 1, 10, 3, help="检索最相关的文档数量", key="top_k_slider_1")
            with col2:
                temperature = st.slider("温度", 0.0, 1.0, 0.1, 0.1,
                                        help="控制生成文本的随机性，值越低越确定", key="temp_slider_1")
            with col3:
                max_tokens = st.slider("最大长度", 100, 2000, 512, 50,
                                       help="生成文本的最大长度", key="max_tokens_slider_1")
            with col4:
                use_enhanced = st.checkbox("增强模式", value=True,
                                           help="使用增强搜索和问答功能", key="enhanced_mode_1")

        # 查询输入区
        st.subheader("💭 输入问题")

        # 从session state获取查询
        if "query" in st.session_state and st.session_state.query:
            query = st.text_area(
                "请输入你的问题：",
                value=st.session_state.query,
                height=120,
                placeholder="例如：什么是机器学习？或者 Python有什么特点？",
                key="question_input_area_1"
            )
        else:
            query = st.text_area(
                "请输入你的问题：",
                height=120,
                placeholder="例如：什么是机器学习？或者 Python有什么特点？",
                key="question_input_area_1"
            )

        # 查询按钮
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_clicked = st.button("🔍 搜索答案", type="primary", use_container_width=True,
                                       key="search_answer_btn_1")
        with col2:
            enhanced_clicked = st.button("🚀 增强搜索", use_container_width=True, key="enhanced_search_btn_1")
        with col3:
            clear_clicked = st.button("🗑️ 清除", use_container_width=True, key="clear_btn_1")

        if clear_clicked:
            for key in ["query", "last_result", "last_query", "search_history"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # 处理增强搜索
        if enhanced_clicked and query.strip():
            use_enhanced = True
            search_clicked = True

        # 示例问题
        st.subheader("💡 快速提问")
        example_cols = st.columns(4)
        examples = [
            "什么是Python？",
            "机器学习是什么？",
            "解释一下RAG技术",
            "深度学习有什么用？"
        ]

        for i, example in enumerate(examples):
            if example_cols[i].button(example, use_container_width=True, key=f"example_btn_{i}"):
                st.session_state.query = example
                st.rerun()

        # 处理查询
        if search_clicked and query.strip():
            with st.spinner("🔍 正在搜索和生成答案..."):
                try:
                    start_time = time.time()

                    # 确保集合存在
                    success, message = ensure_collection_exists(rag_engine)
                    if not success:
                        st.error(f"❌ {message}")
                        return

                    # 选择使用哪种问答方法
                    if use_enhanced:
                        # 使用增强版问答
                        result = rag_engine.ask_enhanced(
                            question=query,
                            collection_name="ai_knowledge_base",
                            top_k=top_k,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            use_enhanced_search=use_enhanced
                        )
                        search_type = "增强搜索"
                    else:
                        # 使用普通问答
                        result = rag_engine.ask(
                            question=query,
                            collection_name="ai_knowledge_base",
                            top_k=top_k,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        search_type = "普通搜索"

                    response_time = time.time() - start_time

                    # 确保result不为None
                    if result is None:
                        result = {
                            "question": query,
                            "answer": "抱歉，查询失败，没有获取到结果。",
                            "sources": [],
                            "sources_count": 0,
                            "search_success": False,
                            "response_time": response_time,
                            "search_type": search_type
                        }

                    # 确保result有必要的字段
                    if "response_time" not in result:
                        result["response_time"] = response_time
                    if "search_type" not in result:
                        result["search_type"] = search_type

                    # 保存结果到session state
                    st.session_state.last_result = result
                    st.session_state.last_query = query

                    # 添加到搜索历史
                    if "search_history" not in st.session_state:
                        st.session_state.search_history = []
                    st.session_state.search_history.append({
                        "query": query,
                        "time": time.strftime("%H:%M:%S"),
                        "type": search_type
                    })

                    # 显示结果
                    st.markdown("---")
                    st.subheader("💡 AI回答")

                    # 美化答案显示
                    answer = result.get("answer", "无答案")
                    st.markdown(f'{answer}', unsafe_allow_html=True)

                    # 显示参考文档
                    sources_count = result.get("sources_count", 0)
                    if sources_count > 0:
                        st.subheader("📚 参考文档")

                        sources = result.get("sources", [])
                        for i, source in enumerate(sources):
                            with st.expander(
                                    f"文档 {i + 1} | 相关度: {source.get('score', 0):.1%}" if 'score' in source else f"文档 {i + 1}",
                                    expanded=(i == 0)):
                                # 文档卡片
                                content = source.get('content_preview', source.get('content', '无内容'))
                                st.markdown(f"""

                                    内容:

                                    {content}

                                """, unsafe_allow_html=True)

                                if source.get("metadata"):
                                    with st.expander("查看元数据"):
                                        st.json(source["metadata"])

                    # 显示统计信息
                    with st.expander("📊 性能统计", expanded=False):
                        cols = st.columns(4)
                        cols[0].metric("响应时间", f"{result.get('response_time', response_time):.2f}s")
                        cols[1].metric("参考文档", sources_count)
                        cols[2].metric("搜索类型", result.get("search_type", "未知"))
                        cols[3].metric("状态", "✅ 成功" if result.get("search_success", True) else "⚠️ 有限")

                        # 显示使用的模型
                        st.write("**使用的模型:**")
                        st.write(f"- LLM: {result.get('model_used', 'deepseek-coder:6.7b')}")
                        st.write(f"- 嵌入模型: {result.get('embedding_model', 'all-MiniLM-L6-v2')}")

                except Exception as e:
                    st.error(f"❌ 查询失败: {str(e)}")
                    import traceback
                    with st.expander("查看错误详情"):
                        st.code(traceback.format_exc())

        # 显示上一次的结果
        elif "last_result" in st.session_state and st.session_state.last_result is not None:
            result = st.session_state.last_result
            query = st.session_state.get("last_query", "")

            if result:
                st.markdown("---")
                st.subheader("📋 上次查询结果")

                if query:
                    st.info(f"**查询:** {query}")

                # 显示答案
                answer = result.get("answer", "无答案")
                st.markdown(f'{answer}', unsafe_allow_html=True)

                # 显示统计信息
                with st.expander("📊 性能统计", expanded=False):
                    cols = st.columns(4)
                    cols[0].metric("响应时间", f"{result.get('response_time', 0):.2f}s")
                    cols[1].metric("参考文档", result.get("sources_count", 0))
                    cols[2].metric("搜索类型", result.get("search_type", "未知"))
                    cols[3].metric("状态", "✅ 成功" if result.get("search_success", True) else "⚠️ 有限")

                    # 显示使用的模型
                    st.write("**使用的模型:**")
                    st.write(f"- LLM: {result.get('model_used', 'deepseek-coder:6.7b')}")
                    st.write(f"- 嵌入模型: {result.get('embedding_model', 'all-MiniLM-L6-v2')}")

    with tab2:
        st.header("🔍 文档搜索")

        if rag_engine:
            # 搜索功能
            st.subheader("搜索文档")

            search_query = st.text_input("搜索关键词:", placeholder="输入要搜索的内容...", key="doc_search_query_2")
            search_top_k = st.slider("返回结果数", 1, 20, 5, key="search_top_k_slider_2")

            col1, col2 = st.columns(2)
            with col1:
                search_btn = st.button("🔍 开始搜索", use_container_width=True, key="doc_search_btn_2")
            with col2:
                enhanced_search_btn = st.button("🚀 增强搜索", use_container_width=True, key="doc_enhanced_search_btn_2")

            if (search_btn or enhanced_search_btn) and search_query.strip():
                with st.spinner("正在搜索..."):
                    try:
                        # 确保集合存在
                        success, message = ensure_collection_exists(rag_engine)
                        if not success:
                            st.error(message)
                            return

                        # 选择搜索方法
                        if enhanced_search_btn:
                            search_results = rag_engine.search_enhanced(
                                query=search_query,
                                collection_name="ai_knowledge_base",
                                top_k=search_top_k
                            )
                            search_type = "增强搜索"
                        else:
                            search_results = rag_engine.search(
                                query=search_query,
                                collection_name="ai_knowledge_base",
                                top_k=search_top_k
                            )
                            search_type = "普通搜索"

                        # 显示搜索结果
                        if search_results.get("success") and search_results.get("count", 0) > 0:
                            st.success(f"✅ 找到 {search_results['count']} 个相关文档 ({search_type})")

                            for i, (doc, metadata) in enumerate(zip(
                                    search_results.get("documents", []),
                                    search_results.get("metadatas", [])
                            )):
                                with st.expander(f"文档 {i + 1}", expanded=(i == 0)):
                                    # 显示文档内容
                                    st.write(doc)

                                    # 显示元数据和分数
                                    cols = st.columns(3)
                                    if i < len(search_results.get("distances", [])):
                                        score = max(0, 1.0 - search_results["distances"][i])
                                        cols[0].metric("相关度", f"{score:.1%}")

                                    if metadata:
                                        with cols[1]:
                                            st.write("元数据:")
                                            st.json(metadata)
                        else:
                            st.warning("⚠️ 没有找到相关文档")

                    except Exception as e:
                        st.error(f"搜索失败: {str(e)}")

            # 显示搜索历史
            if "search_history" in st.session_state and st.session_state.search_history:
                st.subheader("📋 搜索历史")

                history_df = []
                for item in st.session_state.search_history[-10:]:  # 只显示最近10条
                    history_df.append({
                        "时间": item["time"],
                        "查询": item["query"][:50] + ("..." if len(item["query"]) > 50 else ""),
                        "类型": item.get("type", "普通")
                    })

                if history_df:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(history_df), use_container_width=True, key="history_df_2")

                    if st.button("清空历史记录", type="secondary", key="clear_history_btn_2"):
                        del st.session_state.search_history
                        st.rerun()

    with tab3:
        st.header("📊 系统监控")

        if rag_engine:
            # 显示引擎信息
            st.subheader("🖥️ 引擎信息")
            col1, col2, col3 = st.columns(3)
            col1.metric("引擎类型", type(rag_engine).__name__)
            col2.metric("状态", "✅ 运行中")
            col3.metric("LLM模型", "deepseek-coder:6.7b")

            # 知识库信息
            st.subheader("📚 知识库信息")

            try:
                collections = rag_engine.list_collections()

                if collections:
                    # 创建选项卡显示每个集合
                    collection_tabs = st.tabs(collections[:3])  # 最多显示3个集合

                    for idx, collection_name in enumerate(collections[:3]):
                        with collection_tabs[idx]:
                            try:
                                info = rag_engine.get_collection_info(collection_name)
                                if "error" not in info:
                                    st.metric("文档数量", info.get('count', 0))
                                    if info.get('metadata'):
                                        st.write("元数据:")
                                        st.json(info['metadata'])
                                else:
                                    st.warning(f"获取信息失败: {info.get('error')}")
                            except Exception as e:
                                st.warning(f"获取集合信息异常: {str(e)}")
                else:
                    st.info("暂无知识库，请在侧边栏创建或添加文档")
            except Exception as e:
                st.warning(f"获取知识库列表失败: {str(e)}")

            # 文件系统信息
            st.subheader("💾 文件系统")

            col1, col2, col3 = st.columns(3)

            with col1:
                core_dir = project_root / "core"
                if core_dir.exists():
                    file_count = len([f for f in os.listdir(core_dir) if f.endswith('.py')])
                    st.success(f"✅ core目录 ({file_count}个.py文件)")
                else:
                    st.error("❌ core目录不存在")

            with col2:
                data_dir = project_root / "complete_rag_data"
                if data_dir.exists():
                    try:
                        file_count = len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])
                        st.success(f"✅ 数据目录 ({file_count}个文件)")
                    except:
                        st.success("✅ 数据目录")
                else:
                    st.info("📁 数据目录（未创建）")

            with col3:
                models_dir = project_root / "models"
                if models_dir.exists():
                    st.success("✅ models目录")
                else:
                    st.warning("⚠️ models目录（未创建）")


if __name__ == "__main__":
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 初始化session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    # 运行主应用
    main()