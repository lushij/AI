"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/30
    Time:上午6:20
    To change this template use File | Settings | File Templates
"""
# 测试模型
import torch
from LLM import LLM  # 你的模型文件
from BPE_Tokenizer import BPE_Tokenizer  # 你的分词器
import os
from IPython.display import display, Markdown  # 用于漂亮的显示
import config as conf

# ================= 🔧 必须配置的部分 =================

MODEL_PATH = "sft_llama.pth"  # 你的权重文件名
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🖥️ 运行设备: {DEVICE}")


def load_model():
    print("⏳ 正在初始化分词器...")
    try:
        tokenizer = BPE_Tokenizer()
        tokenizer.load("chinese_tokenizer.json")
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return None, None

    print(f"⏳ 正在加载模型权重: {MODEL_PATH} ...")

    # 初始化模型
    model = LLM(conf).to(DEVICE)

    # 加载权重
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict, strict=False)  # strict=False 可以容忍微小的key差异
            print("✅ 模型加载成功！准备就绪。")
        except Exception as e:
            print(f"❌ 权重加载报错: {e}")
            print("💡 提示：请检查 MODEL_CONFIG 里的层数、维度是否与训练时一致。")
    else:
        print("❌ 找不到 .pth 文件，请检查路径。")

    model.eval()  # 切换到评估模式 (关闭 Dropout)
    return model, tokenizer


def count_parameters(model):
    # 计算所有 requires_grad=True 的参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型总参数量: {total_params:,}")
    print(f"💾 估算模型权重大小: {total_params * 4 / (1024 * 1024):.2f} MB (FP32)")


def generate_response(model, tokenizer, question, max_new_tokens=100, temperature=0.7):
    """
    生成回复的核心函数
    """
    if model is None: return "模型未加载"

    # 1. 严格按照 SFT 训练时的格式拼接
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    input_ids = tokenizer.encode(prompt)

    # 转为 Tensor
    x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    # 获取特殊 Token ID
    end_token_id = tokenizer.special_tokens.get("<|end|>", 0)

    # 用于显示的 buffer
    generated_text = ""

    # 开始生成
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 前向传播
            logits = model(x)

            # 取最后一个 token 的 logits
            next_token_logits = logits[:, -1, :]

            # --- 采样策略 (Temperature Sampling) ---
            if temperature > 0:
                # 温度越高，越随机；温度越低，越保守
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪搜索 (Greedy Search) - 永远选概率最大的
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # 获取 token 数值
            next_token_item = next_token.item()

            # 遇到结束符，停止
            if next_token_item == end_token_id:
                break

            # 拼接到输入序列中，作为下一次的上下文
            x = torch.cat([x, next_token], dim=1)

            # 解码当前字符
            word = tokenizer.decode([next_token_item])
            generated_text += word

    return generated_text


def chat_display(question):
    # 朴素版显示，确保你能看见字
    print(f"\n👤 User: {question}")

    response = generate_response(model, tokenizer, question, temperature=0.1)  # 记得保持低温度

    print(f"> 🤖 Assistant: {response}")
    print("-" * 40)

if __name__ == '__main__':

    # 执行加载
    model, tokenizer = load_model()
    count_parameters(model)
    # 测试问题列表
    test_questions = [
        "你好，你是谁？",
        "请把'Good morning'翻译成中文。",
        "1+1等于几？",
        "写一首关于春天的古诗。",
        "如何学习深度学习？"
    ]
    for q in test_questions:
        chat_display(q)
    while True:
        q = input("请输入问题 (输入 q 退出): ")
        if q.lower() == 'q':
            break
        chat_display(q)
