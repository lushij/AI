"""
    Created by PyCharm
    User:lushiji
    Date:2026/1/25
    Time:下午7:27
    To change this template use File | Settings | File Templates
"""


import torch
import torch.nn as nn
import math
class ManualEmbedding(nn.Module):
    """
    纯手写 Embedding 层
    本质：维护一个可学习的大矩阵，并支持通过索引取出对应的行。
    """
    def __init__(self, vocab_size=1000, d_model=64):
        """
        :param vocab_size:词个数
        :param d_model: 维度
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02) #GPT-2用的就是
    def forward(self, idx):
        """
        操作 self.weight[idx]:
        #   这是 PyTorch/NumPy 的“高级索引”功能。
        #   它不是矩阵乘法，它是“物理搬运”。
        #   逻辑：如果 idx 里是 5，就去 self.weight 的第 5 行把整行数据拷贝出来。
        :param idx:
        :return:对应权重矩阵idx行的内容
        """
        return self.weight[idx]



class PositionEncoding(nn.Module):
    """
    实现正弦位置编码 (Sinusoidal Position Encoding)
    注意：这里不需要 vocab_size，而是需要 max_len (句子最大长度)
    """

    def __init__(self, d_model=64, max_len=1000):
        super().__init__()
        even_i = torch.arange(0, d_model, 2).float()

        # 2. 计算分母 (使用 log 空间计算更稳定，数学上等价于 10000^(2i/d_model))
        div_term = torch.exp(even_i * -(math.log(10000.0) / d_model))
        # 3. 建立位置索引 [0, 1, ... max_len-1] -> shape: [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 4. 创建 PE 矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 5. 填充正弦余弦
        # 偶数列 (0, 2, 4...) 使用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数列 (1, 3, 5...) 使用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        # 6. 注册为 buffer
        # 作用：1. 把它加入模型 state_dict 随模型保存
        #       2. 不会把它视为参数(parameter)，不会被梯度更新
        #       3. 模型 .cuda() 时，它会自动跟着去 GPU
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Embedding 层的输出，形状 [Batch_Size, Seq_Len, d_model]
        """
        # 从 PE 表里切出前 Seq_Len 行，加到 x 上
        # x.size(1) 就是当前句子的实际长度
        # 广播机制会自动处理 Batch 维度
        return x + self.pe[:x.size(1), :]


# --- 测试 ---
if __name__ == "__main__":
    d_model = 64
    max_len = 100
    pos_enc = PositionEncoding(d_model=d_model, max_len=max_len)

    # 模拟 Embedding 输出: Batch=2, Len=10, Dim=64
    dummy_input = torch.zeros(2, 10, 64)
    output = pos_enc(dummy_input)

    print(f"PE 矩阵形状: {pos_enc.pe.shape}")  # 应为 [100, 64]
    print(f"输出形状: {output.shape}")  # 应为 [2, 10, 64]

    # 验证一下奇偶列不同
    print("\n验证首行数据 (前4位):")
    print(f"Index 0 (偶-Sin): {pos_enc.pe[0, 0]}")  # sin(0) = 0
    print(f"Index 1 (奇-Cos): {pos_enc.pe[0, 1]}")  # cos(0) = 1
    print(f"Index 2 (偶-Sin): {pos_enc.pe[0, 2]}")  # sin(0) = 0