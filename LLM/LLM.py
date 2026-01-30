"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/27
    Time:下午10:39
    To change this template use File | Settings | File Templates
"""
"""
    组装LLM模型的各个模块，以及建造模型的代码
"""
from TransformerBlock import *
# from BPE_Tokenizer import BPE_Tokenizer # Tokenizer 通常不放在模型 nn.Module 里面，建议在外面调用
from ManualEmbedding import ManualEmbedding
import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 1. Embedding 层
        self.embedding = ManualEmbedding(vocab_size=config.vocab_size, d_model=config.d_model)

        # 2. 共享组件：旋转位置编码 (RoPE)
        # RoPE 是没有参数的，所以所有层共用一个实例完全没问题，省内存
        self.rotate = Rotate(
            head_model=config.d_model // config.num_heads,
            max_len=config.max_len
        )

        # 3. 堆叠 Transformer Block
        self.transformer_blocks = nn.ModuleList()
        for _ in range(config.num_layers):

            # A. 造一个 RMSNorm
            block_rms = RMSNorm(d_model=config.d_model)

            # B. 造一个 Attention (把共享的 RoPE 传进去)
            block_attn = CausalSelfAttention(
                d_model=config.d_model,
                n_heads=config.num_heads,
                rotatelayer=self.rotate, # 注入共享的 RoPE
                max_len=config.max_len
            )

            # C. 造一个 FFN (SwiGLU)
            block_ffn = SwiGLU(d_model=config.d_model, d_ff=config.d_ff)

            # D. 组装成 Block
            block = TransformerBlock(
                rmsnorm=block_rms,
                catt_mask_rotate=block_attn,
                ffn=block_ffn
            )
            self.transformer_blocks.append(block)

        # 4. 最后的归一化 (保持和 Block 内部一致，用 RMSNorm)
        self.final_norm = RMSNorm(d_model=config.d_model)

        # 5. 输出层 (LM Head)
        self.output_linear = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        # x shape: [Batch, Seq_Len] (整数索引)

        # 1. Embedding
        x = self.embedding(x)  # [B, T, d_model]

        # 2. 通过所有 Transformer 层
        for block in self.transformer_blocks:
            x = block(x)

        # 3. 最后的归一化
        x = self.final_norm(x) # [B, T, d_model]

        # 4. 映射到词表
        logits = self.output_linear(x)  # [B, T, vocab_size]

        # ★★★ 关键修正：千万不要在这里做 Softmax ★★★
        # 训练时 CrossEntropyLoss 会帮你做。
        # 推理时(Generate)我们在外面手动做。
        return logits

    @torch.no_grad()  # 推理模式，不需要算梯度
    def generate(self, idx, max_new_tokens):
        """
        idx: 当前的 token 序列，形状 [B, T] (整数索引)
        max_new_tokens: 还要往后生成多少个词
        """
        # 循环生成 max_new_tokens 次
        for _ in range(max_new_tokens):
            # 1. 截断输入
            # 如果 idx 变得太长超过了 max_len，必须截断，否则位置编码会报错
            # 只保留最后 max_len 个词
            idx_cond = idx[:, -self.config.max_len:]

            # 2. 前向传播
            # 获取 logits, 形状 [B, T, vocab_size]
            logits = self(idx_cond)

            # 3. 只关心最后一个时间步 (预测下一个词)
            # [B, 1, vocab_size]
            logits = logits[:, -1, :]

            # 4. 算概率 (Softmax)
            probs = F.softmax(logits, dim=-1)

            # 5. 采样 (Sampling)
            # 这里简单起见，我们取概率最大的那个词 (Greedy Search)
            # 也可以用 torch.multinomial(probs, num_samples=1) 进行随机采样
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # [B, 1]

            # 6. 拼接 (把它连到屁股后面)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx





# --- 测试代码 ---
if __name__ == '__main__':

    # 1. 实例化配置

    # 2. 打印看看参数对不对
    print("-" * 40)
    print(f"正在初始化模型...")
    print(f"层数: {config.num_layers}")
    print(f"维度: {config.d_model}")
    print(f"头数: {config.num_heads}")
    print(f"最大序列长度: {config.max_len}")
    print(f"词表: {config.vocab_size}")
    print(f"dropout: {config.dropout}")
    print(f"esp: {config.esp}")
    print("-" * 40)

    # 3. 实例化模型
    model = LLM(config)

    # 4. 构造假数据测试
    # batch_size = 2, seq_len = 16
    # 这里的输入是 token ID (整数)，范围是 [0, vocab_size)
    x = torch.randint(0, config.vocab_size, (2, 16))

    # 5. 前向传播
    y = model(x)

    # 6. 验证输出
    print(f"输入形状: {x.shape}")  # [2, 16]
    print(f"输出形状: {y.shape}")  # [2, 16, 3200] (3200是vocab_size)

    # 7. 统计参数量 (这是工业界常用的检查步骤)
    total_params = sum(p.numel() for p in model.parameters())
    print("-" * 40)
    print(f"模型总参数量: {total_params / 1e6:.2f} M (百万)")
    print("-" * 40)
    # ... (之前的代码) ...

    print("-" * 40)
    print("正在尝试生成文本 (随机说话)...")

    # 给它一个开头: 假设 0 是开始符号，或者随便给个 [0]
    start_context = torch.zeros((1, 1), dtype=torch.long)  # [Batch=1, Seq=1]

    # 让它往下编 20 个词
    generated_ids = model.generate(start_context, max_new_tokens=20)

    print(f"生成后的 ID 序列: {generated_ids.tolist()}")
    print("-" * 40)