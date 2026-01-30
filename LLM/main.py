"""
    Created by PyCharm
    User: lushiji
    Date: 2026/1/28
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条库 
import json
# 导入你的模块
from TransformerBlock import *
from LLM import LLM
import config
from BPE_Tokenizer import BPE_Tokenizer
# from torchinfo import summary # 推荐用这个看结构

# ==========================================
# 1. 定义数据集类 (关键！)
# ==========================================
class TextDataset(Dataset):
    def __init__(self, text_path, tokenizer, max_len):
        """
        :param text_path: 文本文件路径 (例如 'shakespeare.txt')
        :param tokenizer: 你的 BPE 分词器实例
        :param max_len: 模型的上下文窗口长度 (config.max_len)
        """
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        # 1. 读取所有文本
        print(f"正在读取文件: {text_path} ...")
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 2. 全文分词 (Encode) -> 变成一长串整数
        # 注意：这一步如果文本很大，会比较慢
        print("正在进行全文 Tokenization (这可能需要一点时间)...")
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"数据加载完毕! 总 Token 数: {len(self.tokens)}")

    def __len__(self):
        # 能够切出多少个长度为 max_len 的样本
        # 比如总长 100，max_len 10，能切 90 个 (步长为1的话)
        # 为了训练效率，我们通常步长设为 max_len
        return len(self.tokens) // self.max_len

    def __getitem__(self, idx):
        # 确定切片的起始位置
        start_idx = idx * self.max_len
        end_idx = start_idx + self.max_len
        
        # 获取输入 x (长度 max_len)
        # 这里的切片要小心越界，通常我们会丢弃最后一点点不够长的数据
        chunk = self.tokens[start_idx : end_idx + 1] # 多取 1 个，为了做错位
        
        if len(chunk) < self.max_len + 1:
            # 如果不够长了（例如最后一段），补0或者随机截取
            # 这里简单处理：直接返回随机的一段 (Robust做法)
            rand_start = torch.randint(0, len(self.tokens) - self.max_len - 1, (1,)).item()
            chunk = self.tokens[rand_start : rand_start + self.max_len + 1]

        # 构造错位数据
        x = chunk[:-1]  # 前 max_len 个
        y = chunk[1:]   # 后 max_len 个 (对应的下一个词)
        
        return x, y





class SFTDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ignore_index = -100  # PyTorch 默认忽略的 Loss 索引

        # 加载数据
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. 提取文本
        instruction = item['instruction']
        input_text = item.get('input', '')
        output_text = item['output']

        # 2. 构造 Prompt (分两部分)
        # Part A: 提问部分 (模型能看到，但不需要预测)
        if input_text:
            prompt = f"<|user|>\n{instruction}\n输入：{input_text}\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"

        # Part B: 回答部分 (模型需要预测)
        answer = f"{output_text}<|end|>"

        # 3. 分别编码 (这是为了精确找到分界线！)
        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer)

        # 4. 拼接 input_ids (完整的对话)
        input_ids = prompt_ids + answer_ids

        # 5. 构造 labels (关键步骤！！！)
        # 提问部分填 -100 (不算 Loss)，回答部分填真实的 token ID
        labels = [self.ignore_index] * len(prompt_ids) + answer_ids

        # 6. 截断 (Truncate)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        # 7. 填充 (Padding)
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            # input 填 0 (假设 pad_id=0)，labels 填 -100
            input_ids = input_ids + [0] * pad_len
            labels = labels + [self.ignore_index] * pad_len

        # 8. 转 Tensor
        # 注意错位预测：x 是 0 到 N-1, y 是 1 到 N
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)

        return x, y














# ==========================================
# 2. 主程序
# ==========================================
if __name__ == '__main__':
    # --- 配置与设备 ---
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # --- 1. 实例化分词器 ---
    # 假设你的 BPE_Tokenizer 有 load 方法或者重新训练
    # 这里假设你已经有了训练好的模型文件，或者直接用原始数据初始化
    tokenizer = BPE_Tokenizer(vocab_size=config.vocab_size)
    # 如果 tokenizer 需要先训练，请先在这里调用 tokenizer.train(text)
    # 或者 tokenizer.load("shakespeare_tokenizer.json")
    tokenizer.load("shakespeare_tokenizer.json")

    # --- 2. 加载数据 ---
    # # 你的文本文件叫 'input.txt'
    dataset = TextDataset('input.txt', tokenizer, config.max_len)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # batch_size 根据显存调整

    # --- 3. 初始化模型 ---
    model = LLM(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # --- 4. 优化器与损失 ---
    # 学习率建议：LLM 通常用 3e-4 或 1e-4，0.01 太大了容易不收敛
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充部分的 Loss,sft 特殊处理
    # --- 5. 训练循环 ---
    epochs = 5  # 跑几轮
    loss_history = []

    model.train() # 开启训练模式 (Dropout生效)
    
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            # logits: [Batch, Seq_Len, Vocab_Size]
            logits = model(x)
            
            # ★★★ 关键：变形计算 Loss ★★★
            # CrossEntropyLoss 需要 input 是 [N, C], target 是 [N]
            # B*T 就是 N (总共预测了多少个字)
            B, T, C = logits.shape
            loss = criterion(logits.view(B*T, C), y.view(B*T))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸，LLM 训练必备)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # 记录与打印
            loss_history.append(loss.item())
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Mean Loss: {epoch_loss / len(dataloader):.4f}")
        
        # --- 每轮结束后生成一段话看看效果 ---
        # 切换到 eval 模式
        model.eval()
        start_tokens = torch.zeros((1, 1), dtype=torch.long).to(device) # 给一个起始符
        generated = model.generate(start_tokens, max_new_tokens=50)
        # 解码成文本
        print(f"Epoch {epoch+1} 生成演示: \n{tokenizer.decode(generated[0].tolist())}\n")
        model.train() # 切回训练模式

    # --- 6. 画 Loss 图 ---
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

    # --- 7. 保存模型 ---
    torch.save(model.state_dict(), "mini_llama.pth")
    print("模型已保存!")