"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/29
    Time:下午4:13
    To change this template use File | Settings | File Templates
"""
"""
这个是预训练

"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import os
import wandb  # 引入 wandb


import config as conf
from LLM import LLM
from BPE_Tokenizer import BPE_Tokenizer


# ==========================================
# 1. 定义预训练数据集 (专门处理纯文本)
# ==========================================
import pickle  # 需要引入这个库用来存缓存


class PretrainDataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 自动生成缓存文件名 (比如 wiki_clean.txt -> wiki_clean.bin)
        cache_path = txt_path.replace(".txt", ".bin")

        # 【策略 1】如果有缓存，直接加载 (1秒搞定)
        if os.path.exists(cache_path):
            print(f"🚀 发现缓存文件: {cache_path}，正在极速加载...")
            try:
                with open(cache_path, 'rb') as f:
                    self.tokens = pickle.load(f)
                print(f"✅ 加载完毕，总 Token 数: {len(self.tokens)}")
                return
            except Exception:
                print("⚠️ 缓存文件损坏，重新处理...")

        # 【策略 2】没有缓存，开始切块分词
        print(f"📚 正在读取语料: {txt_path} ...")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到文件: {txt_path}")

        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()  # 先读入内存

        print(f"⏳ 开始分词 (总计 {len(text)} 字符)...")

        # --- 核心修改：分块处理，避免假死 ---
        self.tokens = []
        # 按照 10000 字符一块进行切分
        chunk_size = 10000
        total_chunks = len(text) // chunk_size + 1

        # 使用 tqdm 显示进度条
        for i in tqdm(range(0, len(text), chunk_size), desc="Tokenizing", unit="chunk"):
            chunk = text[i: i + chunk_size]
            if chunk:
                ids = tokenizer.encode(chunk)
                self.tokens.extend(ids)

        # 加上结束符
        self.tokens += [tokenizer.special_tokens.get("<|end|>", 0)]
        print(f"✅ 分词完毕，总 Token 数: {len(self.tokens)}")

        # --- 核心修改：保存缓存 ---
        print(f"💾 正在保存缓存到 {cache_path} (下次直接加载)...")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.tokens, f)
        print("✅ 缓存保存成功！")

    def __len__(self):
        if len(self.tokens) <= self.max_len: return 0
        return (len(self.tokens) - 1) // self.max_len

    def __getitem__(self, idx):
        start_idx = idx * self.max_len
        end_idx = start_idx + self.max_len + 1
        chunk = self.tokens[start_idx: end_idx]

        chunk = torch.tensor(chunk, dtype=torch.long)
        return chunk[:-1], chunk[1:]


def warm_cos_keep(current_step, total_step, max_lr, min_lr, warmup_steps, hold_steps):
    """
    :param current_step: 当前步数
    :param total_step: 总步数
    :param max_lr: 最大学习率
    :param min_lr: 最小学习率
    :param warmup_steps: 预热步数
    :param hold_steps: 保持步数
    :return:
    """
    # 升温
    if current_step < warmup_steps:
        # 线性增长
        lr = max_lr * (current_step / warmup_steps)
    # 余弦退热
    elif current_step < total_step - hold_steps:
        # 我们定义 decay_step，表示"衰减阶段已经跑了多少步"
        # 当 current_step == warmup_steps 时，decay_step 为 0，cos(0)=1，完美衔接 max_lr
        decay_step = current_step - warmup_steps

        decay_total = total_step - warmup_steps - hold_steps

        lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1 + math.cos(decay_step / decay_total * math.pi))
    # 保持
    else:
        lr = min_lr
    return lr


# ==========================================
# 3. 预训练主逻辑
# ==========================================
def train_pretrain():
    # --- 配置 ---
    TXT_PATH = "wiki_clean.txt"  # 你的小说/百科文件路径
    SAVE_PATH = "pretrain_llama.pth"  # 保存的文件名
    TOKENIZER_PATH = "chinese_tokenizer.json"

    BATCH_SIZE = 8  # 根据显存调整
    EPOCHS = 1  # 预训练语料如果很大，跑 1 轮通常就够了
    MAX_LR = 3e-4  # 预训练学习率可以稍微大一点点
    MIN_LR = 3e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化 WandB ---
    wandb.init(project="Mini-LLaMA-Pretrain", name="Pretrain-Run-1", config={
        "type": "Pretrain", "corpus": TXT_PATH, "model_size": "57M"
    })

    # --- 加载 ---
    tokenizer = BPE_Tokenizer()
    tokenizer.load(TOKENIZER_PATH)
    print(f"✅ 分词器加载成功，实际词表大小: {len(tokenizer.vocab)}")
    conf.vocab_size = len(tokenizer.vocab)  # 自动对齐词表

    model = LLM(conf).to(device)  # 从头随机初始化
    print(f"🔥 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    dataset = PretrainDataset(TXT_PATH, tokenizer, max_len=conf.max_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') # 混合精度

    # --- 调度器参数 ---
    total_steps = len(dataloader) * EPOCHS
    warmup_steps = int(total_steps * 0.05)
    hold_steps = int(total_steps * 0.1)

    print(f"🚀 开始预训练! 总Batch数: {len(dataloader)}")

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch_idx, (x, y) in enumerate(loop):
            global_step = epoch * len(dataloader) + batch_idx

            # 1. 更新 LR
            current_lr = warm_cos_keep(global_step, total_steps, MAX_LR, MIN_LR, warmup_steps, hold_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 2. 训练
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits.view(-1, conf.vocab_size), y.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # 3. WandB Log
            wandb.log({"loss": loss.item(), "lr": current_lr, "epoch": epoch + 1})
            loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

    # 保存
    torch.save(model.state_dict(), SAVE_PATH)
    wandb.finish()
    print(f"🎉 预训练完成！权重已保存为 {SAVE_PATH}")


if __name__ == "__main__":
    train_pretrain()