"""
    User: lushiji
    Date: 2026/1/28
    描述: SFT (Supervised Fine-Tuning) 专用训练脚本 (WandB + Pretrain版)
    更新: 加入梯度累积、梯度裁剪、Dropout配置
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import math
import wandb

# 导入你的模块
import config as conf
from LLM import LLM
from BPE_Tokenizer import BPE_Tokenizer

# ==========================================
# 1. 定义 SFT 数据集 (保持不变)
# ==========================================
class SFTDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ignore_index = -100

        print(f"正在加载数据: {json_file} ...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item.get('input', '')
        output_text = item['output']

        if input_text:
            prompt = f"<|user|>\n{instruction}\n输入：{input_text}\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{instruction}\n<|assistant|>\n"
        answer = f"{output_text}<|end|>"

        prompt_ids = self.tokenizer.encode(prompt)
        answer_ids = self.tokenizer.encode(answer)
        input_ids = prompt_ids + answer_ids
        labels = [self.ignore_index] * len(prompt_ids) + answer_ids

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            labels = labels + [self.ignore_index] * pad_len

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)
        return x, y

# ==========================================
# 2. 手动调度器 (Warmup + Cosine + Keep)
# ==========================================
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
# 3. SFT 主训练逻辑
# ==========================================
def train_sft():
    # --- A. 配置 ---
    TOKENIZER_PATH = "chinese_tokenizer.json"
    DATA_PATH = r"alpaca_zh_51k/alpaca_data_51k.json"
    SAVE_PATH = "sft_llama.pth"
    PRETRAIN_PATH = "pretrain_llama.pth"

    BATCH_SIZE = 8       # 单次前向传播的样本数
    ACCUM_STEPS = 4      # ★★★ 新增：梯度累积步数 (等效 Batch Size = 8 * 4 = 32)
    EPOCHS = 3
    MAX_LR = 1e-4
    MIN_LR = 1e-5

    # ★★★ 新增：Dropout 配置 (防止过拟合) ★★★
    if hasattr(conf, 'dropout'):
        conf.dropout = 0.1
        print(f"🛡️ Dropout 已设置为: {conf.dropout}")
    else:
        print("⚠️ Config 中未找到 dropout 字段，请检查 config.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- B. 初始化 WandB ---
    wandb.init(project="Mini-LLaMA-SFT", name="Alpaca-Run-Accum", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS, # 记录累积步数
        "effective_batch_size": BATCH_SIZE * ACCUM_STEPS, # 记录等效 Batch Size
        "lr": MAX_LR,
        "dropout": getattr(conf, 'dropout', 0.0)
    })

    # --- C. 加载模型与分词器 ---
    tokenizer = BPE_Tokenizer()
    tokenizer.load(TOKENIZER_PATH)
    conf.vocab_size = len(tokenizer.vocab)

    model = LLM(conf).to(device)

    if os.path.exists(PRETRAIN_PATH):
        print(f"📥 发现预训练权重: {PRETRAIN_PATH}，正在加载...")
        model.load_state_dict(torch.load(PRETRAIN_PATH), strict=True)
        print("🎉 预训练权重加载成功！")
    else:
        print("⚠️ 未找到预训练权重，将从头开始！")

    # --- D. 优化器与混合精度 ---
    dataset = SFTDataset(DATA_PATH, tokenizer, max_len=conf.max_len)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler('cuda')

    # --- E. 调度器参数 ---
    total_steps = len(dataloader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    hold_steps = int(total_steps * 0.1)

    print(f"🔥 开始 SFT | 总步数: {total_steps} | 累积步数: {ACCUM_STEPS}")

    model.train()
    optimizer.zero_grad() # ★★★ 循环开始前先清零一次

    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (x, y) in enumerate(loop):
            global_step = epoch * len(dataloader) + batch_idx

            # 1. 调度器 (每一步都调整 LR，保持平滑)
            current_lr = warm_cos_keep(global_step, total_steps, MAX_LR, MIN_LR, warmup_steps, hold_steps)
            for param_group in optimizer.param_groups: param_group['lr'] = current_lr

            # 2. 前向传播
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast('cuda'):
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y.view(B * T))

            # ★★★ 关键点 1: Loss 除以累积步数 ★★★
            # 这样累积多次后的梯度总和才是正确的平均值
            loss = loss / ACCUM_STEPS

            # 3. 反向传播 (此时只累积梯度，不更新参数)
            scaler.scale(loss).backward()

            # ★★★ 关键点 2: 只有当凑够了 ACCUM_STEPS 时，才更新参数 ★★★
            if (batch_idx + 1) % ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):

                # --- 梯度裁剪 (Gradient Clipping) ---
                # 必须在 unscale 之后，step 之前进行
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 更新参数
                scaler.step(optimizer)
                scaler.update()

                # 清空梯度
                optimizer.zero_grad()

            # 4. Log 到 WandB (还原 loss 数值以便观察)
            # 注意：显示的 Loss 是单步的 Loss，所以要乘回去
            wandb.log({"loss": loss.item() * ACCUM_STEPS, "lr": current_lr, "epoch": epoch+1})
            loop.set_postfix(loss=loss.item() * ACCUM_STEPS, lr=f"{current_lr:.2e}")

        # --- F. 测试生成 ---
        print("-" * 30)
        print("🤖 测试生成能力:")
        model.eval()
        test_prompt = "<|user|>\n你好\n<|assistant|>\n"
        input_ids = tokenizer.encode(test_prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            for _ in range(50):
                logits = model(input_tensor[:, -conf.max_len:])[:, -1, :]
                logits = logits / 0.8 # Temperature
                v, _ = torch.topk(logits, 20)
                logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                if next_token.item() == tokenizer.special_tokens.get("<|end|>", 0): break
                input_tensor = torch.cat((input_tensor, next_token), dim=1)

            res = tokenizer.decode(input_tensor[0].tolist())
            print(f"Response: {res}")
            wandb.log({"test_sample": wandb.Html(f"<p>{res}</p>")})

        print("-" * 30)
        model.train()

    wandb.finish()
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"🎉 SFT 完成！模型已保存为 {SAVE_PATH}")

if __name__ == "__main__":
    train_sft()